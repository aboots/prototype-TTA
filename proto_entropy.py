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
            
            # --- Entropy Thresholding ---
            if self.entropy_threshold is not None:
                # Calculate Softmax Entropy of the logits (confidence measure)
                # High entropy = uncertain prediction
                probs = logits.softmax(dim=1)
                entropy_vals = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
                
                # Create mask: 1 if reliable (entropy < threshold), 0 if unreliable
                reliable_mask = (entropy_vals < self.entropy_threshold).float()
                
                # Check if we have ANY reliable samples
                if reliable_mask.sum() == 0:
                    # If no samples are reliable, skip update but return results
                    return logits, min_distances, similarities
            else:
                reliable_mask = torch.ones(logits.size(0), device=logits.device)
            
            proto_class_identity = self.model.prototype_class_identity.to(logits.device)
            proto_identities = proto_class_identity.argmax(dim=1)

        # 3. Flatten/Max over sub-prototypes
        if similarities.dim() == 3:
            sim_scores, _ = similarities.max(dim=2) # (B, P)
        else:
            sim_scores = similarities # (B, P)
        
        # 4. Masking: Focus only on prototypes of the predicted class
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        
        # Apply reliability mask (broadcasting to prototypes)
        # We only want to optimize for reliable samples.
        # reliable_mask shape: (B,) -> (B, 1) to match (B, P)
        sample_weights = reliable_mask.unsqueeze(1)
        
        masked_sims = sim_scores * target_mask # (B, P)
        
        # 5. Bipolar Sharpening: Map [-1, 1] -> [0, 1]
        # s = -1 => p = 0
        # s =  0 => p = 0.5 (Max Entropy / Ambiguity)
        # s = +1 => p = 1
        # We assume sim_scores are strictly within [-1, 1].
        # Sometimes numerical errors give slightly outside, so we clamp.
        masked_sims = torch.clamp(masked_sims, min=-1.0, max=1.0)
        proto_probs = (masked_sims + 1.0) / 2.0
        
        # Clamp slightly to avoid log(0)
        eps = 1e-6
        proto_probs = torch.clamp(proto_probs, min=eps, max=1-eps)

        # 6. Minimize Binary Entropy
        # This forces proto_probs towards 0 OR 1.
        # Consequently, it forces similarities towards -1 OR +1.
        # It suppresses values near 0 (ambiguous noise).
        entropy = -(proto_probs * torch.log(proto_probs) + 
                   (1 - proto_probs) * torch.log(1 - proto_probs))
        
        # Average over the target prototypes
        # Note: For non-target prototypes (masked to 0), the similarity is 0.
        # 0 similarity maps to p=0.5, which has MAXIMUM entropy.
        # We do NOT want to minimize entropy for non-target classes (that would force them to -1 or 1).
        # So we multiply by target_mask to only minimize entropy for the PREDICTED class.
        
        masked_entropy = entropy * target_mask
        
        # Weighted mean over target prototypes
        # If reliable_mask is 0 for a sample, its contribution becomes 0.
        weighted_loss = masked_entropy * sample_weights
        
        # Normalize by (number of active prototypes * number of reliable samples)
        # Avoid division by zero
        denom = (target_mask * sample_weights).sum() + 1e-8
        loss = weighted_loss.sum() / denom

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
