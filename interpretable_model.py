import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from toolkits import apply_radial_mask

class InterpretableModel(nn.Module):
    def __init__(self, 
                 mask_generator,
                 classifier,
                 num_classes=10,  # Default to 10, set dynamically for MedMNIST
                 radial_radius=3,
                 radial_decay=0.5,
                 upper_mask_level_threshold=0.8,
                 enable_radial_mask_noise=False,
                 radial_mask_noise_range=0.1):
        super().__init__()
        
        # Store the provided models
        self.mask_generator = mask_generator
        self.classifier = classifier
        
        self.num_classes = num_classes
        self.radial_radius = radial_radius
        self.radial_decay = radial_decay
        self.upper_mask_level_threshold = upper_mask_level_threshold
        self.enable_radial_mask_noise = enable_radial_mask_noise
        self.radial_mask_noise_range = radial_mask_noise_range
        
    def forward(self, x):
        # Generate mask using the mask generator
        soft_mask = self.mask_generator(x)
        
        # Apply radial mask
        radial_mask = apply_radial_mask(soft_mask, radius=self.radial_radius, decay_factor=self.radial_decay)
        
        # Add noise to radial mask during training if enabled
        if self.enable_radial_mask_noise and self.training:
            # Generate uniform noise in range [0, radial_mask_noise_range]
            noise = torch.rand_like(radial_mask) * self.radial_mask_noise_range
            radial_mask = radial_mask + noise
            radial_mask = torch.clamp(radial_mask, 0, 1)
        
        # Create binary mask for pixels above threshold
        binary_mask = (radial_mask > self.upper_mask_level_threshold).float()
        
        # MedMNIST compatibility: handle both grayscale (1 channel) and RGB (3 channel)
        input_channels = x.size(1)
        if input_channels == 3:
            # For RGB images, expand masks to 3 channels
            radial_mask_expanded = radial_mask.repeat(1, 3, 1, 1)
            binary_mask_expanded = binary_mask.repeat(1, 3, 1, 1)
        else:
            # For grayscale images, use masks as is
            radial_mask_expanded = radial_mask
            binary_mask_expanded = binary_mask
        
        # Apply mask to input - INVERTED: now 1 means full masking, 0 means no masking
        # For binary mask, we use it directly (1 means fully masked)
        # For radial mask, we multiply by (1 - mask_expanded)
        masked_x = x * (1 - radial_mask_expanded)
        
        # Apply binary mask on top - fully mask pixels above threshold
        masked_x = masked_x * (1 - binary_mask_expanded)
        
        # Extract features for both masked and unmasked inputs
        # For MedMNIST, classifier should be ResNet18WithMLP or similar
        unmasked_mlp_features = self.classifier.get_features(x)
        masked_mlp_features = self.classifier.get_features(masked_x)
        
        # Get predictions
        unmasked_logits = self.classifier(x)
        masked_logits = self.classifier(masked_x)
        
        return {
            'mask': soft_mask,
            'soft_mask': soft_mask,
            'radial_mask': radial_mask,
            'binary_mask': binary_mask,
            'masked_input': masked_x,
            'masked_x': masked_x,
            'unmasked_logits': unmasked_logits,
            'masked_logits': masked_logits,
            'unmasked_mlp_features': unmasked_mlp_features,
            'masked_mlp_features': masked_mlp_features
        }
    
    def _get_features(self, x):
        return self.classifier.get_features(x) 