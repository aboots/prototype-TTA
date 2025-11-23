import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class GaussianNoise(object):
    """Add Gaussian noise to an image."""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddCorruptions:
    """Wrapper to add various corruptions."""
    def __init__(self, corruption_type='gaussian_noise', severity=1):
        self.corruption_type = corruption_type
        self.severity = severity
        
    def __call__(self, img):
        # Assuming img is a PIL Image or Tensor depending on where we insert this
        # Ideally we insert this after ToTensor
        if self.corruption_type == 'gaussian_noise':
            # Severity determines standard deviation
            std = 0.05 * self.severity 
            noise = torch.randn_like(img) * std
            return torch.clamp(img + noise, 0, 1)
        
        # Add more corruptions here (e.g., shot noise, impulse noise, etc.)
        # For now, we start with simple Gaussian noise as requested
        
        return img

def get_corrupted_transform(img_size, mean, std, corruption_type=None, severity=1):
    """
    Returns a transform pipeline that includes corruption if specified.
    """
    base_transforms = [
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]
    
    if corruption_type:
        base_transforms.append(AddCorruptions(corruption_type, severity))
        
    # Normalization must happen LAST, after noise addition
    base_transforms.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(base_transforms)

