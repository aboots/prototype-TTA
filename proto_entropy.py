from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoEntropy(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 alpha_target=1.0, alpha_separation=0.0, alpha_coherence=0.0,
                 use_prototype_importance=False, use_confidence_weighting=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.alpha_target = alpha_target
        self.alpha_separation = alpha_separation
        self.alpha_coherence = alpha_coherence
        self.use_prototype_importance = use_prototype_importance
        self.use_confidence_weighting = use_confidence_weighting
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, min_distances, similarities = self.forward_and_adapt(x)

        return outputs, min_distances, similarities

    def reset(self):
        # Only reset model parameters, NOT optimizer state
        # Preserves Adam momentum/variance for effective episodic updates
        self.model.load_state_dict(self.model_state, strict=True)
    
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
            proto_class_identity = self.model.prototype_class_identity.to(logits.device)
            proto_identities = proto_class_identity.argmax(dim=1)

        # 3. Flatten/Max over sub-prototypes for main loss
        if similarities.dim() == 3:
            sim_scores, _ = similarities.max(dim=2)
        else:
            sim_scores = similarities

        # 4. Masking: Target and Non-target prototypes
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()
        nontarget_mask = 1.0 - target_mask

        # ========== PART A: Target Entropy Loss ==========
        masked_sims = sim_scores * target_mask
        masked_sims = torch.clamp(masked_sims, min=-1.0, max=1.0)
        proto_probs = (masked_sims + 1.0) / 2.0

        eps = 1e-6
        proto_probs = torch.clamp(proto_probs, min=eps, max=1-eps)

        # Minimize Binary Entropy for target prototypes
        entropy = -(proto_probs * torch.log(proto_probs) + 
                   (1 - proto_probs) * torch.log(1 - proto_probs))
        
        # --- NEW: Prototype Importance Weighting ---
        if self.use_prototype_importance:
            # Get the last_layer weights: [num_classes, num_prototypes]
            # For each sample, get the weights for the predicted class
            last_layer_weights = self.model.last_layer.weight  # [num_classes, num_prototypes]
            
            # Get weights for predicted classes: [batch_size, num_prototypes]
            class_weights = last_layer_weights[pred_class]  # [B, P]
            
            # Normalize weights to [0, 1] range per sample (softmax-like but preserve relative importance)
            # Use absolute values since negative weights also indicate importance
            importance_weights = torch.abs(class_weights)
            
            # Normalize to sum to 1 for each sample (only for target prototypes)
            importance_weights = importance_weights * target_mask
            importance_weights = importance_weights / (importance_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weight the entropy by prototype importance
            weighted_entropy = entropy * target_mask * importance_weights
            loss_per_sample = weighted_entropy.sum(dim=1)  # [B]
        else:
            # Original: uniform weighting across all target prototypes
            masked_entropy = entropy * target_mask
            loss_per_sample = masked_entropy.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)  # [B]
        
        # --- NEW: Confidence Weighting ---
        if self.use_confidence_weighting:
            # Calculate prediction confidence (max probability)
            with torch.no_grad():
                probs = logits.softmax(dim=1)
                confidence = probs.max(dim=1)[0]  # [B]
            
            # Weight the loss by confidence
            # High confidence -> adapt more, Low confidence -> adapt less
            loss_target = (loss_per_sample * confidence).mean()
        else:
            loss_target = loss_per_sample.mean()

        # ========== PART B: Separation Loss (Push non-target prototypes to -1) ==========
        # For non-target prototypes, we want similarities to be close to -1 (dissimilar)
        # Map to [0, 1] and push toward 0
        nontarget_sims = sim_scores * nontarget_mask
        nontarget_sims = torch.clamp(nontarget_sims, min=-1.0, max=1.0)
        nontarget_probs = (nontarget_sims + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]
        nontarget_probs = torch.clamp(nontarget_probs, min=eps, max=1-eps)
        
        # Minimize the probability (push toward 0, meaning similarity toward -1)
        # Using binary cross-entropy with target=0
        separation_loss = -torch.log(1 - nontarget_probs) * nontarget_mask
        loss_separation = separation_loss.sum(dim=1) / (nontarget_mask.sum(dim=1) + 1e-8)
        loss_separation = loss_separation.mean()

        # ========== PART C: Sub-Prototype Coherence Loss ==========
        loss_coherence = 0.0
        if similarities.dim() == 3:
            # similarities: [Batch, Prototypes, Sub-prototypes]
            # For target prototypes, minimize variance across sub-prototypes
            target_similarities = similarities * target_mask.unsqueeze(-1)  # [B, P, K]
            
            # Calculate variance across sub-prototypes (dim=2) for each prototype
            # Only consider target prototypes
            mean_sub = target_similarities.sum(dim=2, keepdim=True) / (similarities.shape[2] + 1e-8)
            variance_sub = ((target_similarities - mean_sub) ** 2).sum(dim=2)  # [B, P]
            
            # Only count target prototypes
            coherence_loss = variance_sub * target_mask
            loss_coherence = coherence_loss.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)
            loss_coherence = loss_coherence.mean()

        # ========== Total Loss ==========
        # Balance the three components using instance attributes
        loss = self.alpha_target * loss_target + self.alpha_separation * loss_separation
        if isinstance(loss_coherence, torch.Tensor):
            loss = loss + self.alpha_coherence * loss_coherence

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, min_distances, similarities


class ProtoEntropyEATA(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, entropy_threshold=0.4):
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
        # Only reset model parameters, NOT optimizer state
        # Preserves Adam momentum/variance for effective episodic updates
        self.model.load_state_dict(self.model_state, strict=True)
    
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
            
            # --- EATA-style thresholding ---
            num_classes = logits.shape[1]
            adaptive_threshold = self.entropy_threshold * torch.log(torch.tensor(num_classes, device=logits.device).float())
            
            probs = logits.softmax(dim=1)
            entropy_vals = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
            reliable_mask = (entropy_vals < adaptive_threshold).float()
            
            if reliable_mask.sum() == 0:
                return logits, min_distances, similarities
            
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

        # --- PART A: Target Loss ---
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

        # --- PART B: Negative Margin Loss ---
        neg_margin = 0.2 
        neg_violation = F.relu(sim_scores - neg_margin)
        
        # Apply masks
        neg_loss_map = neg_violation * nontarget_mask * sample_weights
        
        # Count how many negative prototypes contribute to the loss
        num_neg_prototypes = nontarget_mask.sum(dim=1).mean()
        denom_neg = (sample_weights.sum() * num_neg_prototypes) + 1e-8
        
        loss_neg = neg_loss_map.sum() / denom_neg
        
        # --- Total Loss ---
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
