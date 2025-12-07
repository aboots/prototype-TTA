from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoEntropy(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, entropy_threshold=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.entropy_threshold = entropy_threshold
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, min_distances, similarities = self.forward_and_adapt(x)

        return outputs, min_distances, similarities

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


    @torch.enable_grad()
    def forward_and_adapt(self, x):
        # 1. Forward Pass
        logits, min_distances, similarities = self.model(x)

        # 2. Identify Target Class (Pseudo-label)
        with torch.no_grad():
            pred_class = logits.argmax(dim=1)
            
            # --- Logic for thresholding (Kept from your V2) ---
            if self.entropy_threshold is not None:
                num_classes = logits.shape[1]
                # Fix: Ensure we don't multiply by 0 if threshold is 0, or handle None correctly
                adaptive_threshold = self.entropy_threshold * torch.log(torch.tensor(num_classes, device=logits.device).float())
                
                probs = logits.softmax(dim=1)
                entropy_vals = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
                reliable_mask = (entropy_vals < adaptive_threshold).float()
                
                if reliable_mask.sum() == 0:
                    return logits, min_distances, similarities
            else:
                reliable_mask = torch.ones(logits.size(0), device=logits.device)
            
            proto_class_identity = self.model.prototype_class_identity.to(logits.device)
            proto_identities = proto_class_identity.argmax(dim=1)

        # 3. Flatten/Max over sub-prototypes
        if similarities.dim() == 3:
            sim_scores, _ = similarities.max(dim=2) 
        else:
            sim_scores = similarities 

        # 4. Masks
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        nontarget_mask = 1.0 - target_mask
        sample_weights = reliable_mask.unsqueeze(1) 

        # --- PART A: Target Loss (Same as V1) ---
        masked_sims = sim_scores * target_mask 
        masked_sims = torch.clamp(masked_sims, min=-1.0, max=1.0)
        proto_probs = (masked_sims + 1.0) / 2.0
        eps = 1e-6
        proto_probs = torch.clamp(proto_probs, min=eps, max=1-eps)

        entropy = -(proto_probs * torch.log(proto_probs) + 
                   (1 - proto_probs) * torch.log(1 - proto_probs))
        
        target_loss_map = entropy * target_mask * sample_weights
        
        # Normalize Target Loss by number of reliable SAMPLES
        denom_target = sample_weights.sum() + 1e-8
        loss_target = target_loss_map.sum() / denom_target

        # --- PART B: Negative Margin Loss (The Fix) ---
        neg_margin = 0.2 
        neg_violation = F.relu(sim_scores - neg_margin)
        
        # Apply masks
        neg_loss_map = neg_violation * nontarget_mask * sample_weights
        
        # NORMALIZATION FIX:
        # We must divide by (Num_Samples * Num_Negative_Prototypes)
        # Otherwise this term is 100x larger than loss_target
        
        # Count how many negative prototypes contribute to the loss
        num_neg_prototypes = nontarget_mask.sum(dim=1).mean() # approx (P-1)
        denom_neg = (sample_weights.sum() * num_neg_prototypes) + 1e-8
        
        loss_neg = neg_loss_map.sum() / denom_neg
        
        # --- Total Loss ---
        # Now both losses are on the same scale (per-element average)
        # You can add a weight to neg_loss if needed, e.g., 0.5 * loss_neg
        loss = loss_target + loss_neg

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, min_distances, similarities
# Helper functions
def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with ProtoEntropy adaptation."""
    # train mode, because ProtoEntropy optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what ProtoEntropy updates
    model.requires_grad_(False)
    # configure norm for ProtoEntropy updates: enable grad + force batch statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
