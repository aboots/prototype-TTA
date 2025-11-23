from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

class ProtoEntropy(nn.Module):
    """Adapts a ProtoViT model by maximizing the similarity of the nearest prototypes.
    
    Instead of entropy minimization, this uses a 'denoising' objective.
    Assumption: Noise reduces the cosine similarity between image patches and prototypes.
    Objective: Modify the backbone features to INCREASE the similarity score of the 
    top-k matched prototypes, effectively pulling the feature representation closer 
    to the learned clean prototypes.
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


@torch.enable_grad()
def forward_and_adapt_proto(x, model, optimizer):
    """Forward and adapt model on batch of data using Prototype Similarity Maximization."""
    
    # 1. Forward pass
    # values: (Batch, Num_Prototypes, Sub_Patches) e.g., (B, 2000, 4)
    logits, min_distances, values = model(x)
    
    # 2. Calculate Loss: Maximize Similarity of Top-K Prototypes
    # We assume 'values' represents cosine similarity scores.
    
    # Aggregate sub-patch scores to get total prototype activation
    # shape: (Batch, Num_Prototypes)
    proto_activations = values.sum(dim=-1)
    
    # Find the Top-K strongest prototypes for each image
    # We assume these are the "correct" prototypes and the noise has just lowered their score.
    # We want to boost them back up.
    # K=10 is a reasonable heuristic (roughly 1 class worth of prototypes)
    k = 10 
    top_k_scores, _ = proto_activations.topk(k, dim=1)
    
    # Loss = Negative Sum of Top-K Scores (Maximization)
    loss = -top_k_scores.mean()
    
    # 3. Backward and Update
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
