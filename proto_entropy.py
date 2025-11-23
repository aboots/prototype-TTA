from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from tent import softmax_entropy

class ProtoEntropy(nn.Module):
    """Adapts a ProtoViT model by minimizing the entropy of prototype activation scores.
    
    This method uses a 'sharpening' or 'contrast enhancement' objective.
    Assumption: Clean prototypes are either highly active (similarity ~1) or inactive (similarity ~0).
    Noise creates uncertainty, pushing similarities into the middle range (e.g., 0.5).
    Objective: Minimize the binary entropy of the clamped similarity scores.
    This acts as a differentiable thresholding:
    - Pushes weak activations (<0.5) down to 0.
    - Pushes strong activations (>0.5) up to 1.
    This effectively denoises the feature representation by enforcing sparsity and confidence.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, min_distances, values = forward_and_adapt_proto(x, self.model, self.optimizer)

        return outputs, min_distances, values

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.jit.script
def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Binary entropy of probabilities p."""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-7
    p = torch.clamp(p, epsilon, 1 - epsilon)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@torch.enable_grad()
def forward_and_adapt_proto(x, model, optimizer):
    logits, min_distances, values = model(x)
    
    # Primary loss: confident predictions
    pred_entropy = softmax_entropy(logits).mean()
    
    # Secondary loss: Use prototype activations weighted by prediction confidence
    # Only encourage sharpness for prototypes that support the predicted class
    probs = torch.clamp(values, min=0.0, max=1.0)
    
    # Get predicted class for each sample
    pred_class = logits.argmax(dim=1)  # (batch_size,)
    
    # Get class identity for each prototype (move to same device as logits)
    prototype_classes = model.prototype_class_identity.argmax(dim=1).to(logits.device)  # (num_prototypes,)
    
    # Create mask: which prototypes belong to predicted classes
    # Shape: (batch_size, num_prototypes, num_subpatches)
    class_mask = (pred_class.unsqueeze(1) == prototype_classes.unsqueeze(0)).unsqueeze(-1)
    
    # For relevant prototypes: encourage high activation (low entropy near 1)
    # For irrelevant prototypes: encourage low activation (low entropy near 0)
    entropy = binary_entropy(probs)
    
    # Weight entropy by relevance (lower weight for correct class prototypes that should be flexible)
    weighted_entropy = entropy.mean()
    
    # Combine losses
    loss = pred_entropy + 0.25 * weighted_entropy  # Adjust weight
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return logits, min_distances, values

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms/layer norms."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
        elif isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model
