import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision.utils import save_image
import sys
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F  # For F.normalize
import json
import time  # Add time module for timing

# Function to apply radial mask to a soft mask
def apply_radial_mask(soft_mask, radius=3, decay_factor=0.2):
    """
    Apply a radial influence pattern to a soft mask.
    
    Args:
        soft_mask: Tensor of shape [B, 1, H, W] containing the original soft mask values
        radius: Integer radius of influence (in pixels)
        decay_factor: Float factor for how quickly the influence decays with distance
        
    Returns:
        Tensor of shape [B, 1, H, W] containing the radial-influenced mask
    """
    batch_size, channels, height, width = soft_mask.shape
    
    # Initialize the output tensor
    radial_mask = torch.zeros_like(soft_mask)
    
    # Create a grid of coordinates for the kernel
    y_coords = torch.arange(-radius, radius+1, device=soft_mask.device)
    x_coords = torch.arange(-radius, radius+1, device=soft_mask.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Calculate distances from center for the kernel
    distances = torch.sqrt(y_grid.float()**2 + x_grid.float()**2)
    
    # Create the radial kernel (1 at center, decaying with distance)
    kernel = torch.exp(-decay_factor * distances)
    kernel = kernel / kernel.sum()  # Normalize
    
    # Reshape kernel for convolution
    kernel = kernel.view(1, 1, 2*radius+1, 2*radius+1)
    
    # Apply convolution to get the radial mask
    radial_mask = F.conv2d(soft_mask, kernel, padding=radius)
    
    # Ensure values are still between 0 and 1
    radial_mask = torch.clamp(radial_mask, 0.0, 1.0)
    
    return radial_mask

class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator, self).__init__()
        # Enhanced U-Net architecture with more filters and layers
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4)  # Add dropout for regularization
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 = 256 (from skip) + 256 (from upconv)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 (from skip) + 128 (from upconv)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 (from skip) + 64 (from upconv)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoder path with skip connections
        upconv3 = self.upconv3(bottleneck)
        concat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(concat3)
        
        upconv2 = self.upconv2(dec3)
        concat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(concat1)
        
        # Output mask with sigmoid activation
        output = torch.sigmoid(self.final(dec1))
        return output

# Define the Classifier Network
class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Multi-layer MLP structure
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def get_features(self, x, layer_idx=1):  # layer_idx=1 for first MLP layer
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        # Extract features from specified MLP layer
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == layer_idx * 2 - 1:  # ReLU after linear layer
                return x
        return x
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x

class InterpretableModel(nn.Module):
    def __init__(self, num_classes=10, radial_radius=3, radial_decay=0.5, upper_mask_level_threshold=0.8):
        super(InterpretableModel, self).__init__()
        self.mask_generator = MaskGenerator()
        self.classifier = Classifier(num_classes)
        self.radial_radius = radial_radius
        self.radial_decay = radial_decay
        self.upper_mask_level_threshold = upper_mask_level_threshold
        
    def forward(self, x):
        # Generate mask using the mask generator
        soft_mask = self.mask_generator(x)
        
        # Apply radial mask instead of threshold-based approach
        radial_mask = apply_radial_mask(soft_mask, radius=self.radial_radius, decay_factor=self.radial_decay)
        
        # Create binary mask for pixels above threshold
        binary_mask = (radial_mask > self.upper_mask_level_threshold).float()
        
        # Expand masks to match input dimensions
        radial_mask_expanded = radial_mask.repeat(1, 3, 1, 1)
        binary_mask_expanded = binary_mask.repeat(1, 3, 1, 1)
        
        # Apply mask to input - INVERTED: now 1 means full masking, 0 means no masking
        # For binary mask, we use it directly (1 means fully masked)
        # For radial mask, we multiply by (1 - mask_expanded)
        masked_x = x * (1 - radial_mask_expanded)
        
        # Apply binary mask on top - fully mask pixels above threshold
        masked_x = masked_x * (1 - binary_mask_expanded)
        
        # Extract MLP features for both masked and unmasked inputs
        unmasked_mlp_features = self.classifier.get_features(x, layer_idx=1)
        masked_mlp_features = self.classifier.get_features(masked_x, layer_idx=1)
        
        # Get predictions
        unmasked_logits = self.classifier(x)
        masked_logits = self.classifier(masked_x)
        
        return {
            'mask': soft_mask,
            'soft_mask': soft_mask,
            'radial_mask': radial_mask,
            'binary_mask': binary_mask,  # Add binary mask to outputs
            'masked_input': masked_x,
            'unmasked_logits': unmasked_logits,
            'masked_logits': masked_logits,
            'unmasked_mlp_features': unmasked_mlp_features,
            'masked_mlp_features': masked_mlp_features
        }
    
    def _get_features(self, x):
        return self.classifier.features(x).flatten(1)

class MaskedLoss(nn.Module):
    def __init__(self, lambda_mask=1.0, lambda_fully_masked=0.05,
                 dynamic_masked_weight_min=1.0, dynamic_masked_weight_max=5.0,
                 lambda_alignment=0.5, upper_mask_level_threshold=0.8):  # Add threshold parameter
        super(MaskedLoss, self).__init__()
        self.lambda_mask = lambda_mask
        self.lambda_fully_masked = lambda_fully_masked
        self.lambda_alignment = lambda_alignment  # Weight for the feature alignment loss
        self.upper_mask_level_threshold = upper_mask_level_threshold  # Store threshold
        
        # Parameters for dynamic masked loss weighting
        self.dynamic_masked_weight_min = dynamic_masked_weight_min
        self.dynamic_masked_weight_max = dynamic_masked_weight_max
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, targets):
        soft_mask = outputs['soft_mask']
        radial_mask = outputs['radial_mask']  # Use radial mask instead of hard_mask
        binary_mask = outputs['binary_mask']  # Use binary mask from model
        mask = outputs['mask']
        unmasked_logits = outputs['unmasked_logits']
        masked_logits = outputs['masked_logits']
        
        # Extract MLP features for alignment loss
        unmasked_mlp_features = outputs['unmasked_mlp_features']
        masked_mlp_features = outputs['masked_mlp_features']
        
        # 1. Calculate per-sample losses for both masked and unmasked inputs
        masked_losses = self.ce_loss(masked_logits, targets)
        unmasked_losses = self.ce_loss(unmasked_logits, targets)
        
        # Calculate the mean losses for comparison
        masked_loss_mean = masked_losses.mean()
        unmasked_loss_mean = unmasked_losses.mean()
        
        # Calculate the gap between masked and unmasked loss
        loss_gap = torch.clamp(masked_loss_mean - unmasked_loss_mean, min=0.0)
        
        # Compute the dynamic weight factor based on the loss gap
        dynamic_weight = torch.clamp(
            1.0 + loss_gap,
            min=self.dynamic_masked_weight_min,
            max=self.dynamic_masked_weight_max
        )
        
        # Apply the dynamic weight to the masked loss
        weighted_masked_loss = dynamic_weight * masked_loss_mean
        
        # Total classification loss combines weighted masked loss and unmasked loss
        classification_loss = weighted_masked_loss + unmasked_loss_mean
        
        # 2. Masking loss - encourage masking as many pixels as possible
        # With inverted interpretation, we want to encourage higher values (closer to 1)
        mask_mean = torch.mean(soft_mask)
        # Add exponential penalty for unmasked pixels to create stronger gradient
        # Now we want to maximize mask_mean, so we penalize (1 - mask_mean)
        masking_loss = self.lambda_mask * ((1 - mask_mean) ** 2)  # Quadratic penalty for unmasked pixels
        
        # 3. Binary loss - encourage values to be either 0 or 1 (perfect entropy)
        # This creates a strong gradient pushing values toward 0 or 1
        # The term radial_mask * (1 - radial_mask) is maximum at 0.5 and minimum at 0 and 1
        # So we want to minimize this term to push values toward 0 or 1
        # Apply binary loss to the radial mask instead of the soft mask
        binary_loss = self.lambda_fully_masked * torch.mean(radial_mask * (1 - radial_mask))
        
        # 4. Feature alignment loss - now using MLP features
        # Normalize the MLP features to compute cosine similarity
        unmasked_mlp_norm = F.normalize(unmasked_mlp_features, p=2, dim=1)
        
        # Detach masked features to prevent gradients from flowing through the masked path
        masked_mlp_norm = F.normalize(masked_mlp_features.detach(), p=2, dim=1)
        
        # Compute cosine similarity between normalized MLP feature vectors
        cosine_similarity = torch.sum(unmasked_mlp_norm * masked_mlp_norm, dim=1)
        
        # Convert similarity to a divergence measure (1 - similarity) and take mean
        alignment_divergence = 1.0 - cosine_similarity.mean()
        
        # Apply weight to the alignment loss
        alignment_loss = self.lambda_alignment * alignment_divergence
        
        # Total loss with alignment term
        total_loss = classification_loss + masking_loss + binary_loss + alignment_loss
        
        # Calculate binary mask metrics (only count fully masked pixels > threshold)
        fully_masked_pixels = (radial_mask > self.upper_mask_level_threshold).float()
        fully_masked_pct = 100 * torch.mean(fully_masked_pixels).item()
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'masked_loss': masked_loss_mean,
            'unmasked_loss': unmasked_loss_mean,
            'masking_loss': masking_loss,
            'fully_masked_loss': binary_loss,
            'alignment_loss': alignment_loss,  # Add the new loss to returned dict
            'alignment_divergence': alignment_divergence,  # Add raw divergence value
            'mask_mean': mask_mean,
            'binary_mask_mean': torch.mean(radial_mask).item(),  # Use radial mask instead of hard_mask
            'fully_masked_pct': fully_masked_pct,
            'dynamic_weight': dynamic_weight
        }

# Helper function to get accuracy
def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return (predicted == labels).sum().item() / labels.size(0) * 100

# Update validate_model function to track dynamic weight
def validate_model(data_loader):
    model.eval()
    
    val_metrics = {
        'masked_correct': 0,
        'unmasked_correct': 0,
        'total_samples': 0,
        'masked_loss': 0.0,
        'unmasked_loss': 0.0,
        'mask_mean': 0.0,
        'binary_mask_mean': 0.0,
        'fully_masked_pct': 0.0,
        'total_loss': 0.0,
        'masking_loss': 0.0,
        'fully_masked_loss': 0.0,
        'dynamic_weight': 0.0,
        'alignment_loss': 0.0,
        'alignment_divergence': 0.0
    }
    
    # Start timing validation
    val_start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss_dict = criterion(outputs, labels)
            
            _, masked_preds = torch.max(outputs['masked_logits'], 1)
            _, unmasked_preds = torch.max(outputs['unmasked_logits'], 1)
            
            batch_size = labels.size(0)
            val_metrics['total_samples'] += batch_size
            val_metrics['masked_correct'] += (masked_preds == labels).sum().item()
            val_metrics['unmasked_correct'] += (unmasked_preds == labels).sum().item()
            
            # Sum losses
            val_metrics['total_loss'] += loss_dict['total_loss'].item() * batch_size
            val_metrics['masked_loss'] += loss_dict['masked_loss'].item() * batch_size
            val_metrics['unmasked_loss'] += loss_dict['unmasked_loss'].item() * batch_size
            val_metrics['masking_loss'] += loss_dict['masking_loss'].item() * batch_size
            val_metrics['fully_masked_loss'] += loss_dict['fully_masked_loss'].item() * batch_size
            val_metrics['dynamic_weight'] += loss_dict['dynamic_weight'].item() * batch_size
            val_metrics['alignment_loss'] += loss_dict['alignment_loss'].item() * batch_size  # Add alignment loss
            val_metrics['alignment_divergence'] += loss_dict['alignment_divergence'].item() * batch_size  # Add divergence
            
            # Sum mask means - both continuous and binary
            val_metrics['mask_mean'] += outputs['soft_mask'].mean().item() * batch_size
            val_metrics['binary_mask_mean'] += outputs['radial_mask'].mean().item() * batch_size
            val_metrics['fully_masked_pct'] += loss_dict['fully_masked_pct'] * batch_size
    
    # Calculate validation time
    val_time = time.time() - val_start_time
    
    # Calculate final metrics
    total = val_metrics['total_samples']
    
    return {
        'masked_acc': 100 * val_metrics['masked_correct'] / total,
        'unmasked_acc': 100 * val_metrics['unmasked_correct'] / total,
        'masked_loss': val_metrics['masked_loss'] / total,
        'unmasked_loss': val_metrics['unmasked_loss'] / total,
        'total_loss': val_metrics['total_loss'] / total,
        'mask_mean': val_metrics['mask_mean'] / total,
        'binary_mask_mean': val_metrics['binary_mask_mean'] / total,
        'masked_pixels_pct': (val_metrics['mask_mean'] / total) * 100,  # Use mask_mean directly (1 = masked)
        'fully_masked_pct': val_metrics['fully_masked_pct'] / total,
        'masking_loss': val_metrics['masking_loss'] / total,
        'fully_masked_loss': val_metrics['fully_masked_loss'] / total,
        'dynamic_weight': val_metrics['dynamic_weight'] / total,
        'alignment_loss': val_metrics['alignment_loss'] / total,  # Add alignment loss average
        'alignment_divergence': val_metrics['alignment_divergence'] / total,  # Add divergence average
        'val_time': val_time  # Add validation time to returned metrics
    }

def train_model():
    print("STARTING TRAINING")
    
    # Initialize metrics tracking
    best_val_masked_acc = 0
    metrics_header = "epoch,lambda_mask,lambda_fully_masked,dynamic_weight,train_total_loss,train_masked_loss,train_unmasked_loss,train_masking_loss,train_fully_masked_loss,train_mask_mean,train_masked_pixels_pct,train_binary_mask_mean,train_fully_masked_pct,train_masked_acc,train_unmasked_acc,val_total_loss,val_masked_loss,val_unmasked_loss,val_mask_mean,val_masked_pixels_pct,val_binary_mask_mean,val_fully_masked_pct,val_masked_acc,val_unmasked_acc,val_masking_loss,val_fully_masked_loss,val_dynamic_weight,train_alignment_loss,train_alignment_divergence,val_alignment_loss,val_alignment_divergence,lambda_alignment\n"
    
    # Simplified to just track metrics in a CSV file
    with open(os.path.join(log_dir, "training_metrics.csv"), "w") as f:
        f.write(metrics_header)
    
    # Create efficiency log file
    with open(os.path.join(log_dir, "efficiency.txt"), "w") as f:
        f.write("Epoch,Training Time (s),Validation Time (s),Total Epoch Time (s)\n")
    
    # Create checkpoints directory
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Initialize timing variables
    total_experiment_time = time.time()
    epoch_times = []
    validation_times = []
    
    # Initially freeze U-Net weights if start_epoch_unet > 0
    if start_epoch_unet > 0:
        for param in model.mask_generator.parameters():
            param.requires_grad = False
        print(f"U-Net weights frozen until epoch {start_epoch_unet}")
    
    for epoch in range(num_epochs):
        # Start timing epoch
        epoch_start_time = time.time()
        
        # Only create directories when needed (for visualization)
        if (epoch + 1) % visualization_interval == 0:
            epoch_train_dir = os.path.join(train_img_dir, f"epoch_{epoch+1}")
            epoch_val_dir = os.path.join(val_img_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_train_dir, exist_ok=True)
            os.makedirs(epoch_val_dir, exist_ok=True)
        
        model.train()
        
        # Unfreeze U-Net weights if we've reached the start epoch
        if epoch == start_epoch_unet:
            for param in model.mask_generator.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch+1}: Unfreezing U-Net weights")
        
        # Update lambda values based on current epoch, considering start epochs
        # Mask loss
        if epoch >= start_epoch_mask:
            # Calculate progress ratio only for epochs after start_epoch
            progress_ratio = (epoch - start_epoch_mask) / max(1, (num_epochs - 1 - start_epoch_mask))
            current_lambda_mask = initial_lambda_mask + (final_lambda_mask - initial_lambda_mask) * progress_ratio
        else:
            current_lambda_mask = 0.0  # No mask loss before start epoch
        criterion.lambda_mask = current_lambda_mask
        
        # Binary mask loss
        if epoch >= start_epoch_fully_masked:
            progress_ratio = (epoch - start_epoch_fully_masked) / max(1, (num_epochs - 1 - start_epoch_fully_masked))
            current_lambda_fully_masked = initial_lambda_fully_masked + (final_lambda_fully_masked - initial_lambda_fully_masked) * progress_ratio
        else:
            current_lambda_fully_masked = 0.0  # No binary mask loss before start epoch
        criterion.lambda_fully_masked = current_lambda_fully_masked
        
        # Dynamic weighting (optional - you can decide if this should honor start epoch too)
        if epoch < start_epoch_dynamic_weight:
            # Force weight to 1.0 before start epoch (no dynamic weighting)
            criterion.dynamic_masked_weight_min = 1.0
            criterion.dynamic_masked_weight_max = 1.0
        else:
            # Restore original dynamic weight range after start epoch
            criterion.dynamic_masked_weight_min = dynamic_masked_weight_min
            criterion.dynamic_masked_weight_max = dynamic_masked_weight_max
        
        train_metrics = {
            'total_loss': 0.0,
            'masked_loss': 0.0,
            'unmasked_loss': 0.0,
            'masking_loss': 0.0,
            'fully_masked_loss': 0.0,
            'alignment_loss': 0.0,  # Add alignment loss
            'alignment_divergence': 0.0,  # Add raw divergence value
            'mask_mean': 0.0,
            'binary_mask_mean': 0.0,
            'fully_masked_pct': 0.0,
            'masked_acc': 0.0,
            'unmasked_acc': 0.0,
            'dynamic_weight': 0.0
        }
        
        # Process mini-batches with tqdm progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, labels) in enumerate(train_loader_tqdm):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss_dict = criterion(outputs, labels)
            
            # Calculate accuracies
            masked_acc = get_accuracy(outputs['masked_logits'], labels)
            unmasked_acc = get_accuracy(outputs['unmasked_logits'], labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # Update metrics
            for key in ['total_loss', 'masked_loss', 'unmasked_loss', 'masking_loss', 
                       'fully_masked_loss', 'mask_mean', 
                       'binary_mask_mean', 'fully_masked_pct', 'dynamic_weight',
                       'alignment_loss', 'alignment_divergence']:  # Add new metrics
                if key in loss_dict:
                    train_metrics[key] += loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
            
            train_metrics['masked_acc'] += masked_acc
            train_metrics['unmasked_acc'] += unmasked_acc
            
            # Update tqdm description with current loss, binary mask metrics, dynamic weight, and alignment
            train_loader_tqdm.set_postfix(
                loss=f"{loss_dict['total_loss'].item():.4f}", 
                m_acc=f"{masked_acc:.2f}%",
                mask=f"{loss_dict['mask_mean'].item()*100:.1f}%",
                masked=f"{loss_dict['fully_masked_pct']:.1f}%",
                visible=f"{100 * torch.mean((outputs['soft_mask'] > 0.8).float()).item():.1f}%",
                dw=f"{loss_dict['dynamic_weight'].item():.2f}",
                align=f"{loss_dict['alignment_divergence'].item():.2f}"  # Add alignment divergence
            )
        
        # Calculate training epoch averages
        batch_count = len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= batch_count
        
        # Calculate masked pixel percentage for training - use mask_mean directly (1 = masked)
        train_masked_pixels_pct = train_metrics['mask_mean'] * 100
        
        # Run validation
        val_metrics = validate_model(val_loader)
        
        # Record validation time
        validation_times.append(val_metrics['val_time'])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log timing information
        with open(os.path.join(log_dir, "efficiency.txt"), "a") as f:
            f.write(f"{epoch+1},{epoch_time - val_metrics['val_time']:.2f},{val_metrics['val_time']:.2f},{epoch_time:.2f}\n")
        
        # Improved terminal output with more details
        print(f"Epoch {epoch+1}/{num_epochs}, λ_mask={current_lambda_mask:.2f}, λ_fully_masked={current_lambda_fully_masked:.4f}")
        print(f"  TRAIN - Dynamic Weight: {train_metrics['dynamic_weight']:.2f}, Val: {val_metrics['dynamic_weight']:.2f}")
        print(f"  TRAIN - Unmasked: Loss={train_metrics['unmasked_loss']:.4f}, Acc={train_metrics['unmasked_acc']:.2f}%")
        print(f"  TRAIN - Masked:   Loss={train_metrics['masked_loss']:.4f}, Acc={train_metrics['masked_acc']:.2f}%, Mask Loss={train_metrics['masking_loss']:.4f}")
        print(f"  TRAIN - Masking:  Soft={train_masked_pixels_pct:.1f}%, Binary={train_metrics['fully_masked_pct']:.1f}%")
        print(f"  TRAIN - Binary Loss: Raw={train_metrics['fully_masked_loss']:.4f}, Weighted={train_metrics['fully_masked_loss']:.4f}")
        print(f"  TRAIN - Alignment: Loss={train_metrics['alignment_loss']:.4f}, Divergence={train_metrics['alignment_divergence']:.4f}")
        print(f"  VAL   - Unmasked: Loss={val_metrics['unmasked_loss']:.4f}, Acc={val_metrics['unmasked_acc']:.2f}%")
        print(f"  VAL   - Masked:   Loss={val_metrics['masked_loss']:.4f}, Acc={val_metrics['masked_acc']:.2f}%, Mask Loss={val_metrics['masking_loss']:.4f}")
        print(f"  VAL   - Masking:  Soft={val_metrics['masked_pixels_pct']:.1f}%, Binary={val_metrics['fully_masked_pct']:.1f}%")
        print(f"  VAL   - Binary Loss: Raw={val_metrics['fully_masked_loss']:.4f}, Weighted={val_metrics['fully_masked_loss']:.4f}")
        print(f"  VAL   - Alignment: Loss={val_metrics['alignment_loss']:.4f}, Divergence={val_metrics['alignment_divergence']:.4f}")
        print(f"  TIME  - Training: {epoch_time - val_metrics['val_time']:.2f}s, Validation: {val_metrics['val_time']:.2f}s, Total: {epoch_time:.2f}s")
        
        # Check if this is the best model so far
        if val_metrics['masked_acc'] > best_val_masked_acc:
            best_val_masked_acc = val_metrics['masked_acc']
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best_model.pth"))
        
        # Save metrics to CSV for plotting later
        with open(os.path.join(log_dir, "training_metrics.csv"), "a") as f:
            f.write(f"{epoch+1},{current_lambda_mask:.4f},{current_lambda_fully_masked:.4f},"
                   f"{train_metrics['dynamic_weight']:.4f},{train_metrics['total_loss']:.6f},"
                   f"{train_metrics['masked_loss']:.6f},{train_metrics['unmasked_loss']:.6f},"
                   f"{train_metrics['masking_loss']:.6f},{train_metrics['fully_masked_loss']:.6f},"
                   f"{train_metrics['mask_mean']:.6f},{train_masked_pixels_pct:.2f},"
                   f"{train_metrics['binary_mask_mean']:.6f},{train_metrics['fully_masked_pct']:.2f},"
                   f"{train_metrics['masked_acc']:.2f},{train_metrics['unmasked_acc']:.2f},"
                   f"{val_metrics['total_loss']:.6f},{val_metrics['masked_loss']:.6f},"
                   f"{val_metrics['unmasked_loss']:.6f},{val_metrics['mask_mean']:.6f},"
                   f"{val_metrics['masked_pixels_pct']:.2f},{val_metrics['binary_mask_mean']:.6f},"
                   f"{val_metrics['fully_masked_pct']:.2f},{val_metrics['masked_acc']:.2f},"
                   f"{val_metrics['unmasked_acc']:.2f},{val_metrics['masking_loss']:.6f},"
                   f"{val_metrics['fully_masked_loss']:.6f},{val_metrics['dynamic_weight']:.4f},"
                   f"{train_metrics['alignment_loss']:.6f},{train_metrics['alignment_divergence']:.6f},"
                   f"{val_metrics['alignment_loss']:.6f},{val_metrics['alignment_divergence']:.6f},"
                   f"{criterion.lambda_alignment:.4f}\n")
        
        # Visualize results only at specified intervals
        if (epoch + 1) % visualization_interval == 0:
            visualize_results(epoch, epoch_train_dir, epoch_val_dir)
        
        # Save model checkpoint only at specified intervals
        if (epoch + 1) % visualization_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"model_epoch_{epoch+1}.pth"))
    
    # Calculate total experiment time
    total_experiment_time = time.time() - total_experiment_time
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "model_final.pth"))
    
    # Calculate and save timing summary
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_validation_time = sum(validation_times) / len(validation_times)
    
    with open(os.path.join(log_dir, "efficiency.txt"), "a") as f:
        f.write("\nSUMMARY\n")
        f.write(f"Total experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)\n")
        f.write(f"Average epoch time: {avg_epoch_time:.2f} seconds\n")
        f.write(f"Average validation time: {avg_validation_time:.2f} seconds\n")
        f.write(f"Average training time per epoch: {avg_epoch_time - avg_validation_time:.2f} seconds\n")
        f.write(f"Total epochs: {num_epochs}\n")
        f.write(f"Visualization interval: Every {visualization_interval} epochs\n")
        f.write(f"Checkpoint interval: Every {visualization_interval} epochs\n")

# Function to visualize results from both train and val sets
def visualize_results(epoch, train_dir, val_dir, num_samples=8):
    model.eval()
    
    # Visualize training samples
    visualize_dataset_samples(train_loader, epoch, num_samples, "train", train_dir)
    
    # Visualize validation samples
    visualize_dataset_samples(val_loader, epoch, num_samples, "val", val_dir)

# Update the visualization function to show both soft and radial masks
def visualize_dataset_samples(data_loader, epoch, num_samples, dataset_type, save_dir):
    # Get some examples
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Select a subset of samples to visualize
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        soft_mask = outputs['soft_mask']
        radial_mask = outputs['radial_mask']
        binary_mask = outputs['binary_mask']  # Get binary mask
        masked_images = outputs['masked_input']
        
        # Get predictions
        _, masked_preds = torch.max(outputs['masked_logits'], 1)
        _, unmasked_preds = torch.max(outputs['unmasked_logits'], 1)
        
        # Move tensors to CPU for visualization
        images = images.cpu()
        soft_mask = soft_mask.cpu()
        radial_mask = radial_mask.cpu()
        binary_mask = binary_mask.cpu()  # Move binary mask to CPU
        masked_images = masked_images.cpu()
        labels = labels.cpu()
        masked_preds = masked_preds.cpu()
        unmasked_preds = unmasked_preds.cpu()
        
        # Create a figure with rows of images - now with 5 columns to show binary mask
        fig, axs = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        
        for i in range(num_samples):
            # Original image with unmasked prediction
            img = images[i].numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # Unnormalize
            axs[i, 0].imshow(img)
            correct = "✓" if unmasked_preds[i] == labels[i] else "✗"
            axs[i, 0].set_title(f'Original: {classes[labels[i]]}\nPred: {classes[unmasked_preds[i]]} {correct}')
            axs[i, 0].axis('off')
            
            # Soft Mask
            soft_mask_mean = soft_mask[i].mean().item()
            axs[i, 1].imshow(soft_mask[i].squeeze(), cmap='viridis')
            axs[i, 1].set_title(f'Soft Mask (1=Masked, 0=Visible)\nMean: {soft_mask_mean:.3f}')
            axs[i, 1].axis('off')
            
            # Radial Mask
            radial_mask_mean = radial_mask[i].mean().item()
            axs[i, 2].imshow(radial_mask[i].squeeze(), cmap='viridis')
            axs[i, 2].set_title(f'Radial Mask (1=Masked, 0=Visible)\nMean: {radial_mask_mean:.3f}')
            axs[i, 2].axis('off')
            
            # Binary Mask (pixels > threshold)
            binary_mask_mean = binary_mask[i].mean().item()
            binary_pct = binary_mask_mean * 100
            axs[i, 3].imshow(binary_mask[i].squeeze(), cmap='binary')
            axs[i, 3].set_title(f'Binary Mask (> {model.upper_mask_level_threshold})\nMasked: {binary_pct:.1f}%')
            axs[i, 3].axis('off')
            
            # Masked image with prediction
            masked_img = masked_images[i].numpy().transpose((1, 2, 0))
            masked_img = masked_img * 0.5 + 0.5  # Unnormalize
            axs[i, 4].imshow(masked_img)
            correct = "✓" if masked_preds[i] == labels[i] else "✗"
            axs[i, 4].set_title(f'Masked\nPred: {classes[masked_preds[i]]} {correct}')
            axs[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"results_samples.png"), dpi=150)
        plt.close(fig)
        
        # Save mask grids - soft, radial, and binary
        soft_mask_grid = torchvision.utils.make_grid(soft_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(soft_mask_grid, os.path.join(save_dir, f"soft_masks_grid.png"))
        
        radial_mask_grid = torchvision.utils.make_grid(radial_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(radial_mask_grid, os.path.join(save_dir, f"radial_masks_grid.png"))
        
        binary_mask_grid = torchvision.utils.make_grid(binary_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(binary_mask_grid, os.path.join(save_dir, f"binary_masks_grid.png"))
        
        # Save masked images grid
        masked_grid = torchvision.utils.make_grid(masked_images, nrow=4, normalize=True)
        torchvision.utils.save_image(masked_grid, os.path.join(save_dir, f"masked_images_grid.png"))

def plot_training_metrics():
    import pandas as pd
    
    # Load metrics from CSV
    metrics_df = pd.read_csv(os.path.join(log_dir, "training_metrics.csv"))
    
    # Create plots
    plt.figure(figsize=(15, 35))  # Increased height for more plots
    
    # Losses plot - Training vs Validation
    plt.subplot(7, 2, 1)
    plt.plot(metrics_df['epoch'], metrics_df['train_masked_loss'], 'b-', label='Train Masked Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_masked_loss'], 'b--', label='Val Masked Loss')
    plt.plot(metrics_df['epoch'], metrics_df['train_unmasked_loss'], 'g-', label='Train Unmasked Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_unmasked_loss'], 'g--', label='Val Unmasked Loss')
    plt.title('Classification Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dynamic weight plot
    plt.subplot(7, 2, 2)
    plt.plot(metrics_df['epoch'], metrics_df['dynamic_weight'], 'r-', label='Train Dynamic Weight')
    plt.plot(metrics_df['epoch'], metrics_df['val_dynamic_weight'], 'r--', label='Val Dynamic Weight')
    plt.title('Dynamic Masked Loss Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss gap vs Dynamic weight
    plt.subplot(7, 2, 3)
    # Calculate the loss gap
    metrics_df['train_loss_gap'] = metrics_df['train_masked_loss'] - metrics_df['train_unmasked_loss']
    plt.scatter(metrics_df['train_loss_gap'], metrics_df['dynamic_weight'], c=metrics_df['epoch'], cmap='viridis')
    plt.colorbar(label='Epoch')
    plt.title('Loss Gap vs Dynamic Weight')
    plt.xlabel('Masked-Unmasked Loss Gap')
    plt.ylabel('Dynamic Weight')
    plt.grid(True, alpha=0.3)
    
    # Masking loss plot
    plt.subplot(7, 2, 4)
    plt.plot(metrics_df['epoch'], metrics_df['train_masking_loss'], 'r-', label='Train Masking Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_masking_loss'], 'r--', label='Val Masking Loss')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_mask'], 'k--', label='Lambda Mask')
    plt.title('Masking Loss & Lambda')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Binary Masked loss plot
    plt.subplot(7, 2, 5)
    plt.plot(metrics_df['epoch'], metrics_df['train_fully_masked_loss'], 'c-', label='Train Binary Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_fully_masked_loss'], 'c--', label='Val Binary Loss')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_fully_masked'], 'k--', label='Lambda Binary')
    plt.title('Binary Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracies plot - Training vs Validation
    plt.subplot(7, 2, 6)
    plt.plot(metrics_df['epoch'], metrics_df['train_masked_acc'], 'b-', label='Train Masked Acc')
    plt.plot(metrics_df['epoch'], metrics_df['val_masked_acc'], 'b--', label='Val Masked Acc')
    plt.plot(metrics_df['epoch'], metrics_df['train_unmasked_acc'], 'g-', label='Train Unmasked Acc')
    plt.plot(metrics_df['epoch'], metrics_df['val_unmasked_acc'], 'g--', label='Val Unmasked Acc')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dynamic weight vs accuracies
    plt.subplot(7, 2, 7)
    plt.plot(metrics_df['dynamic_weight'], metrics_df['train_masked_acc'], 'bo-', label='Train Masked Acc')
    plt.plot(metrics_df['val_dynamic_weight'], metrics_df['val_masked_acc'], 'go-', label='Val Masked Acc')
    for i, txt in enumerate(metrics_df['epoch']):
        plt.annotate(txt, (metrics_df['dynamic_weight'].iloc[i], metrics_df['train_masked_acc'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
    plt.title('Dynamic Weight vs Masked Accuracy')
    plt.xlabel('Dynamic Weight')
    plt.ylabel('Masked Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Soft Masking percentage plot - Training vs Validation
    plt.subplot(7, 2, 8)
    plt.plot(metrics_df['epoch'], metrics_df['train_masked_pixels_pct'], 'm-', label='Train Soft Masked %')
    plt.plot(metrics_df['epoch'], metrics_df['val_masked_pixels_pct'], 'm--', label='Val Soft Masked %')
    plt.title('Soft Masking Percentage')
    plt.xlabel('Epoch')
    plt.ylabel('Masked Pixels (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Binary Masking percentage plot - Training vs Validation
    plt.subplot(7, 2, 9)
    plt.plot(metrics_df['epoch'], metrics_df['train_fully_masked_pct'], 'y-', label='Train Binary Masked %')
    plt.plot(metrics_df['epoch'], metrics_df['val_fully_masked_pct'], 'y--', label='Val Binary Masked %')
    plt.title('Binary Masking Percentage')
    plt.xlabel('Epoch')
    plt.ylabel('Masked Pixels (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Total loss plot
    plt.subplot(7, 2, 10)
    plt.plot(metrics_df['epoch'], metrics_df['train_total_loss'], 'b-', label='Train Total Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_total_loss'], 'b--', label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # All regularization parameters comparison (updated to include alignment)
    plt.subplot(7, 2, 11)
    plt.plot(metrics_df['epoch'], metrics_df['lambda_mask'], 'r-', label='Lambda Mask')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_fully_masked'], 'c-', label='Lambda Binary')
    plt.plot(metrics_df['epoch'], metrics_df['dynamic_weight'], 'g-', label='Dynamic Weight')
    plt.title('Regularization Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Check if alignment metrics exist in the DataFrame
    has_alignment_metrics = 'train_alignment_loss' in metrics_df.columns and 'val_alignment_loss' in metrics_df.columns
    
    # Alignment loss plot (only if metrics exist)
    plt.subplot(7, 2, 12)
    if has_alignment_metrics:
        plt.plot(metrics_df['epoch'], metrics_df['train_alignment_loss'], 'r-', label='Train Alignment Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_alignment_loss'], 'r--', label='Val Alignment Loss')
        if 'lambda_alignment' in metrics_df.columns:
            plt.plot(metrics_df['epoch'], metrics_df['lambda_alignment'], 'k--', label='Lambda Alignment')
        plt.title('Alignment Loss')
    else:
        plt.text(0.5, 0.5, 'Alignment metrics not available', ha='center', va='center')
        plt.title('Alignment Loss (Not Available)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Alignment divergence plot (only if metrics exist)
    plt.subplot(7, 2, 13)
    if has_alignment_metrics and 'train_alignment_divergence' in metrics_df.columns and 'val_alignment_divergence' in metrics_df.columns:
        plt.plot(metrics_df['epoch'], metrics_df['train_alignment_divergence'], 'b-', label='Train Alignment Divergence')
        plt.plot(metrics_df['epoch'], metrics_df['val_alignment_divergence'], 'b--', label='Val Alignment Divergence')
        plt.title('Feature Alignment Divergence')
    else:
        plt.text(0.5, 0.5, 'Alignment divergence metrics not available', ha='center', va='center')
        plt.title('Feature Alignment Divergence (Not Available)')
    plt.xlabel('Epoch')
    plt.ylabel('Divergence (1-cosine_similarity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_metrics.png"), dpi=150)
    plt.close()
    
    # Create a specific plot just for dynamic weight vs loss gap
    plt.figure(figsize=(10, 8))
    metrics_df['val_loss_gap'] = metrics_df['val_masked_loss'] - metrics_df['val_unmasked_loss']
    
    plt.scatter(metrics_df['train_loss_gap'], metrics_df['dynamic_weight'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, alpha=0.7, label='Train')
    plt.scatter(metrics_df['val_loss_gap'], metrics_df['val_dynamic_weight'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, marker='x', alpha=0.7, label='Validation')
    
    for i, txt in enumerate(metrics_df['epoch']):
        plt.annotate(txt, (metrics_df['train_loss_gap'].iloc[i], metrics_df['dynamic_weight'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
        plt.annotate(txt, (metrics_df['val_loss_gap'].iloc[i], metrics_df['val_dynamic_weight'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.colorbar(label='Epoch')
    plt.title('Loss Gap vs Dynamic Weight')
    plt.xlabel('Masked-Unmasked Loss Gap')
    plt.ylabel('Dynamic Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "dynamic_weight_analysis.png"), dpi=150)
    plt.close()
    
    # Create a new plot for alignment divergence vs. masked accuracy (only if metrics exist)
    if has_alignment_metrics and 'train_alignment_divergence' in metrics_df.columns and 'val_alignment_divergence' in metrics_df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(metrics_df['train_alignment_divergence'], metrics_df['train_masked_acc'], 
                    c=metrics_df['epoch'], cmap='viridis', s=100, alpha=0.7, label='Train')
        plt.scatter(metrics_df['val_alignment_divergence'], metrics_df['val_masked_acc'], 
                    c=metrics_df['epoch'], cmap='viridis', s=100, marker='x', alpha=0.7, label='Validation')
        
        for i, txt in enumerate(metrics_df['epoch']):
            plt.annotate(txt, (metrics_df['train_alignment_divergence'].iloc[i], metrics_df['train_masked_acc'].iloc[i]), 
                         textcoords="offset points", xytext=(0,5), ha='center')
            plt.annotate(txt, (metrics_df['val_alignment_divergence'].iloc[i], metrics_df['val_masked_acc'].iloc[i]), 
                         textcoords="offset points", xytext=(0,5), ha='center')
        
        plt.colorbar(label='Epoch')
        plt.title('Alignment Divergence vs Masked Accuracy')
        plt.xlabel('Feature Alignment Divergence')
        plt.ylabel('Masked Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "alignment_analysis.png"), dpi=150)
        plt.close()

# Function to evaluate model on test set
def evaluate_model():
    model.eval()
    
    print("EVALUATION ON TEST SET")
    
    test_metrics = validate_model(test_loader)
    
    # Print improved evaluation summary
    print(f"TEST RESULTS:")
    print(f"  Unmasked: Acc={test_metrics['unmasked_acc']:.2f}%, Loss={test_metrics['unmasked_loss']:.4f}")
    print(f"  Masked:   Acc={test_metrics['masked_acc']:.2f}%, Loss={test_metrics['masked_loss']:.4f}, Mask Loss={test_metrics['masking_loss']:.4f}, Fully Masked Loss={test_metrics['fully_masked_loss']:.4f}, Masked Pixels={test_metrics['masked_pixels_pct']:.2f}%")
    
    return test_metrics

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Get current timestamp for log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = "cifar10"
    
    # Create a more organized log directory structure with timestamp
    log_dir = os.path.join("log", "cifar10", f"{experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Create train and val subdirectories
    train_img_dir = os.path.join(log_dir, "train")
    val_img_dir = os.path.join(log_dir, "val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 64
    num_epochs = 20  # Fewer epochs for quicker results
    learning_rate = 0.001
    
    # Visualization and checkpoint intervals
    visualization_interval = 1  # Generate visualizations every 5 epochs
    checkpoint_interval = 1    # Save checkpoints every epoch
    
    # U-Net weight freezing parameter
    start_epoch_unet = 0  # Epoch to start updating U-Net weights (0 = from beginning)
    
    # Masking loss hyperparameters
    initial_lambda_mask = 1.0 # Start with no masking penalty
    final_lambda_mask = 1.0  # End with strong masking penalty
    start_epoch_mask = 0  # Epoch to start applying mask loss (0 = from beginning)
    
    # Binary loss hyperparameters
    initial_lambda_fully_masked = 0.2  # Start with small fully masked loss weight
    final_lambda_fully_masked = 0.5 # End with stronger fully masked loss weight
    start_epoch_fully_masked = 0  # Epoch to start applying binary loss (0 = from beginning)
    
    # Dynamic masked loss weighting parameters
    dynamic_masked_weight_min = 1.0  # Minimum weight multiplier
    dynamic_masked_weight_max = 2.0  # Maximum weight multiplier
    start_epoch_dynamic_weight = 0  # Epoch to start applying dynamic weighting (0 = from beginning)
    
    # Alignment loss hyperparameters
    initial_lambda_alignment = 0.5 # Start with moderate alignment loss weight
    final_lambda_alignment = 0.5    # End with stronger alignment loss weight
    start_epoch_alignment = 0  # Epoch to start applying alignment loss (0 = from beginning)
    
    # Radial mask hyperparameters
    radial_radius = 1  # Radius of influence for radial mask (in pixels)
    radial_decay = 0.2  # Decay factor for how quickly the influence decays with distance

    # Print experiment configuration
    print(f"Experiment: dynamic_weight_cifar_10")
    print(f"Dataset: CIFAR-10")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"U-Net Start Epoch: {start_epoch_unet} (weights frozen until this epoch)")
    print(f"Initial/Final Lambda Mask: {initial_lambda_mask}/{final_lambda_mask} (Start Epoch: {start_epoch_mask})")
    print(f"Initial/Final Lambda Fully Masked: {initial_lambda_fully_masked}/{final_lambda_fully_masked} (Start Epoch: {start_epoch_fully_masked})")
    print(f"Initial/Final Lambda Alignment: {initial_lambda_alignment}/{final_lambda_alignment} (Start Epoch: {start_epoch_alignment})")
    print(f"Radial Mask Parameters: Radius={radial_radius}, Decay={radial_decay}")
    print(f"Dynamic Masked Weight Range: {dynamic_masked_weight_min} to {dynamic_masked_weight_max} (Start Epoch: {start_epoch_dynamic_weight})")
    print(f"Visualization Interval: Every {visualization_interval} epochs")
    print(f"Checkpoint Interval: Every {checkpoint_interval} epoch")
    print(f"Device: {device}")

    # Save hyperparameters as JSON
    hyperparameters = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "start_epoch_unet": start_epoch_unet,
        "initial_lambda_mask": initial_lambda_mask,
        "final_lambda_mask": final_lambda_mask,
        "start_epoch_mask": start_epoch_mask,
        "initial_lambda_fully_masked": initial_lambda_fully_masked,
        "final_lambda_fully_masked": final_lambda_fully_masked,
        "start_epoch_fully_masked": start_epoch_fully_masked,
        "dynamic_masked_weight_min": dynamic_masked_weight_min,
        "dynamic_masked_weight_max": dynamic_masked_weight_max,
        "start_epoch_dynamic_weight": start_epoch_dynamic_weight,
        "initial_lambda_alignment": initial_lambda_alignment,
        "final_lambda_alignment": final_lambda_alignment,
        "start_epoch_alignment": start_epoch_alignment,
        "radial_radius": radial_radius,
        "radial_decay": radial_decay,
        "visualization_interval": visualization_interval,
        "device": str(device),
        "experiment_name": experiment_name,
        "timestamp": timestamp
    }
    
    # Save hyperparameters to JSON file
    with open(os.path.join(log_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)
    
    print(f"Hyperparameters saved to {os.path.join(log_dir, 'hyperparameters.json')}")

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Suppress download messages
    original_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull  # Redirect output to devnull
        train_dataset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform)

    # Restore normal printing
    sys.stdout = original_stdout

    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Print dataset sizes
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {len(test_dataset)}")

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize the model
    model = InterpretableModel(radial_radius=radial_radius, radial_decay=radial_decay).to(device)
    
    # Load pretrained classifier weights
    pretrained_path = "../../pretrain/model_archtecture_weight_pairs/cifar10_cnn/best_model.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained classifier weights from {pretrained_path}")
        pretrained_state_dict = torch.load(pretrained_path)
        model.classifier.load_state_dict(pretrained_state_dict)
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_path}. Using randomly initialized weights.")
    
    criterion = MaskedLoss(
        lambda_mask=initial_lambda_mask, 
        lambda_fully_masked=initial_lambda_fully_masked,
        lambda_alignment=initial_lambda_alignment,
        dynamic_masked_weight_min=dynamic_masked_weight_min,
        dynamic_masked_weight_max=dynamic_masked_weight_max,
        upper_mask_level_threshold=0.8
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting Interpretable Masking Experiment with Radial Mask and Feature Alignment")
    
    # Initial evaluation before training
    print("\nINITIAL EVALUATION (Before Training)")
    print("=====================================")
    
    # Create initial evaluation directory
    initial_eval_dir = os.path.join(log_dir, "initial_evaluation")
    os.makedirs(initial_eval_dir, exist_ok=True)
    
    # Evaluate on training set
    print("\nEvaluating on training set:")
    train_metrics = validate_model(train_loader)
    print(f"  Unmasked: Acc={train_metrics['unmasked_acc']:.2f}%, Loss={train_metrics['unmasked_loss']:.4f}")
    print(f"  Masked:   Acc={train_metrics['masked_acc']:.2f}%, Loss={train_metrics['masked_loss']:.4f}, Mask Loss={train_metrics['masking_loss']:.4f}")
    print(f"  Masking:  Soft={train_metrics['masked_pixels_pct']:.1f}%, Binary={train_metrics['fully_masked_pct']:.1f}%")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    val_metrics = validate_model(val_loader)
    print(f"  Unmasked: Acc={val_metrics['unmasked_acc']:.2f}%, Loss={val_metrics['unmasked_loss']:.4f}")
    print(f"  Masked:   Acc={val_metrics['masked_acc']:.2f}%, Loss={val_metrics['masked_loss']:.4f}, Mask Loss={val_metrics['masking_loss']:.4f}")
    print(f"  Masking:  Soft={val_metrics['masked_pixels_pct']:.1f}%, Binary={val_metrics['fully_masked_pct']:.1f}%")
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_metrics = validate_model(test_loader)
    print(f"  Unmasked: Acc={test_metrics['unmasked_acc']:.2f}%, Loss={test_metrics['unmasked_loss']:.4f}")
    print(f"  Masked:   Acc={test_metrics['masked_acc']:.2f}%, Loss={test_metrics['masked_loss']:.4f}, Mask Loss={test_metrics['masking_loss']:.4f}")
    print(f"  Masking:  Soft={test_metrics['masked_pixels_pct']:.1f}%, Binary={test_metrics['fully_masked_pct']:.1f}%")
    
    # Save initial metrics to CSV
    initial_metrics_file = os.path.join(initial_eval_dir, "initial_metrics.csv")
    with open(initial_metrics_file, "w") as f:
        f.write("dataset,unmasked_acc,unmasked_loss,masked_acc,masked_loss,masking_loss,masked_pixels_pct,fully_masked_pct\n")
        f.write(f"train,{train_metrics['unmasked_acc']:.2f},{train_metrics['unmasked_loss']:.6f},{train_metrics['masked_acc']:.2f},{train_metrics['masked_loss']:.6f},{train_metrics['masking_loss']:.6f},{train_metrics['masked_pixels_pct']:.2f},{train_metrics['fully_masked_pct']:.2f}\n")
        f.write(f"val,{val_metrics['unmasked_acc']:.2f},{val_metrics['unmasked_loss']:.6f},{val_metrics['masked_acc']:.2f},{val_metrics['masked_loss']:.6f},{val_metrics['masking_loss']:.6f},{val_metrics['masked_pixels_pct']:.2f},{val_metrics['fully_masked_pct']:.2f}\n")
        f.write(f"test,{test_metrics['unmasked_acc']:.2f},{test_metrics['unmasked_loss']:.6f},{test_metrics['masked_acc']:.2f},{test_metrics['masked_loss']:.6f},{test_metrics['masking_loss']:.6f},{test_metrics['masked_pixels_pct']:.2f},{test_metrics['fully_masked_pct']:.2f}\n")
    
    # Visualize initial results
    print("\nGenerating initial visualizations...")
    visualize_results(0, initial_eval_dir, initial_eval_dir)
    
    # Create initial metrics plot
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    datasets = ['Train', 'Val', 'Test']
    unmasked_acc = [train_metrics['unmasked_acc'], val_metrics['unmasked_acc'], test_metrics['unmasked_acc']]
    masked_acc = [train_metrics['masked_acc'], val_metrics['masked_acc'], test_metrics['masked_acc']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, unmasked_acc, width, label='Unmasked')
    plt.bar(x + width/2, masked_acc, width, label='Masked')
    
    plt.ylabel('Accuracy (%)')
    plt.title('Initial Accuracy Comparison')
    plt.xticks(x, datasets)
    plt.legend()
    
    # Loss comparison
    plt.subplot(2, 2, 2)
    unmasked_loss = [train_metrics['unmasked_loss'], val_metrics['unmasked_loss'], test_metrics['unmasked_loss']]
    masked_loss = [train_metrics['masked_loss'], val_metrics['masked_loss'], test_metrics['masked_loss']]
    
    plt.bar(x - width/2, unmasked_loss, width, label='Unmasked')
    plt.bar(x + width/2, masked_loss, width, label='Masked')
    
    plt.ylabel('Loss')
    plt.title('Initial Loss Comparison')
    plt.xticks(x, datasets)
    plt.legend()
    
    # Masking percentages
    plt.subplot(2, 2, 3)
    soft_masked = [train_metrics['masked_pixels_pct'], val_metrics['masked_pixels_pct'], test_metrics['masked_pixels_pct']]
    binary_masked = [train_metrics['fully_masked_pct'], val_metrics['fully_masked_pct'], test_metrics['fully_masked_pct']]
    
    plt.bar(x - width/2, soft_masked, width, label='Soft Masked')
    plt.bar(x + width/2, binary_masked, width, label='Binary Masked')
    
    plt.ylabel('Percentage (%)')
    plt.title('Initial Masking Percentages')
    plt.xticks(x, datasets)
    plt.legend()
    
    # Masking loss
    plt.subplot(2, 2, 4)
    masking_loss = [train_metrics['masking_loss'], val_metrics['masking_loss'], test_metrics['masking_loss']]
    
    plt.bar(x, masking_loss, width)
    
    plt.ylabel('Loss')
    plt.title('Initial Masking Loss')
    plt.xticks(x, datasets)
    
    plt.tight_layout()
    plt.savefig(os.path.join(initial_eval_dir, "initial_metrics.png"), dpi=150)
    plt.close()
    
    print(f"Initial evaluation complete. Results saved to {initial_eval_dir}")
    print("=====================================\n")
    
    train_model()
    evaluate_model()
    plot_training_metrics()
    
    print(f"Experiment complete!")