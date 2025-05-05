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
import torch.nn.functional as F
import json
import time
import pandas as pd

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


class MaskedLoss(nn.Module):
    def __init__(self, lambda_mask=1.0, lambda_fully_masked=0.05,
                 dynamic_masked_weight_min=1.0, dynamic_masked_weight_max=5.0,
                 lambda_alignment=0.5, upper_mask_level_threshold=0.8):
        super(MaskedLoss, self).__init__()
        self.lambda_mask = lambda_mask
        self.lambda_fully_masked = lambda_fully_masked
        self.lambda_alignment = lambda_alignment
        self.upper_mask_level_threshold = upper_mask_level_threshold
        
        # Parameters for dynamic masked loss weighting
        self.dynamic_masked_weight_min = dynamic_masked_weight_min
        self.dynamic_masked_weight_max = dynamic_masked_weight_max
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, targets):
        # Ensure targets are in the correct format (1D tensor)
        targets = targets.squeeze().long()
        
        soft_mask = outputs['soft_mask']
        radial_mask = outputs['radial_mask']
        binary_mask = outputs['binary_mask']
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
        mask_mean = torch.mean(soft_mask)
        masking_loss = self.lambda_mask * ((1 - mask_mean) ** 2)
        
        # 3. Binary loss - encourage values to be either 0 or 1 (perfect entropy)
        binary_loss = self.lambda_fully_masked * torch.mean(radial_mask * (1 - radial_mask))
        
        # 4. Feature alignment loss
        unmasked_mlp_norm = F.normalize(unmasked_mlp_features, p=2, dim=1)
        masked_mlp_norm = F.normalize(masked_mlp_features.detach(), p=2, dim=1)
        cosine_similarity = torch.sum(unmasked_mlp_norm * masked_mlp_norm, dim=1)
        alignment_divergence = 1.0 - cosine_similarity.mean()
        alignment_loss = self.lambda_alignment * alignment_divergence
        
        # Total loss with alignment term
        total_loss = classification_loss + masking_loss + binary_loss + alignment_loss
        
        # Calculate binary mask metrics
        fully_masked_pixels = (radial_mask > self.upper_mask_level_threshold).float()
        fully_masked_pct = 100 * torch.mean(fully_masked_pixels).item()
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'masked_loss': masked_loss_mean,
            'unmasked_loss': unmasked_loss_mean,
            'masking_loss': masking_loss,
            'fully_masked_loss': binary_loss,
            'alignment_loss': alignment_loss,
            'alignment_divergence': alignment_divergence,
            'mask_mean': mask_mean,
            'binary_mask_mean': torch.mean(radial_mask).item(),
            'fully_masked_pct': fully_masked_pct,
            'dynamic_weight': dynamic_weight
        }

def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    # Make sure labels is the right shape (convert from [batch_size, 1] to [batch_size])
    if len(labels.shape) > 1 and labels.shape[1] == 1:
        labels = labels.squeeze()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return (correct / total) * 100

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    print(f"\nValidating on device: {device}")
    print(f"Model parameters device: {next(model.parameters()).device}")
    
    val_metrics = {
        'total_loss': 0.0,
        'masked_loss': 0.0,
        'unmasked_loss': 0.0,
        'masking_loss': 0.0,
        'fully_masked_loss': 0.0,
        'alignment_loss': 0.0,
        'alignment_divergence': 0.0,
        'mask_mean': 0.0,
        'binary_mask_mean': 0.0,
        'fully_masked_pct': 0.0,
        'masked_acc': 0.0,
        'unmasked_acc': 0.0,
        'dynamic_weight': 0.0,
        'masked_pixels_pct': 0.0,
        'validation_time': 0.0
    }
    
    total_batches = len(val_loader)
    total_samples = 0
    total_masked_correct = 0
    total_unmasked_correct = 0
    
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # Ensure labels shape is [batch_size]
            if labels.dim() > 1 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            if batch_idx == 0:  # Print device info for first batch
                print(f"First batch - Images device: {images.device}, Labels device: {labels.device}")
            
            # Forward pass
            outputs = model(images)
            loss_dict = criterion(outputs, labels)
            
            # Calculate accuracies
            _, masked_preds = torch.max(outputs['masked_logits'], 1)
            _, unmasked_preds = torch.max(outputs['unmasked_logits'], 1)
            
            # Count correct predictions
            batch_size = labels.size(0)
            total_samples += batch_size
            total_masked_correct += (masked_preds == labels).sum().item()
            total_unmasked_correct += (unmasked_preds == labels).sum().item()
            
            # Calculate per-batch accuracy for progress display only
            masked_acc = (masked_preds == labels).sum().item() / batch_size * 100
            unmasked_acc = (unmasked_preds == labels).sum().item() / batch_size * 100
            
            # Update metrics
            val_metrics['total_loss'] += loss_dict['total_loss'].item()
            val_metrics['masked_loss'] += loss_dict['masked_loss'].item()
            val_metrics['unmasked_loss'] += loss_dict['unmasked_loss'].item()
            val_metrics['masking_loss'] += loss_dict['masking_loss'].item()
            val_metrics['fully_masked_loss'] += loss_dict['fully_masked_loss'].item()
            val_metrics['alignment_loss'] += loss_dict['alignment_loss'].item()
            val_metrics['alignment_divergence'] += loss_dict['alignment_divergence'].item()
            val_metrics['mask_mean'] += outputs['soft_mask'].mean().item()
            val_metrics['binary_mask_mean'] += outputs['radial_mask'].mean().item()
            val_metrics['fully_masked_pct'] += loss_dict['fully_masked_pct']
            val_metrics['dynamic_weight'] += loss_dict['dynamic_weight'].item()
            val_metrics['masked_pixels_pct'] += (outputs['soft_mask'].mean().item() * 100)
    
    end_time = time.time()
    val_metrics['validation_time'] = end_time - start_time
    
    # Calculate global accuracy
    val_metrics['masked_acc'] = (total_masked_correct / total_samples) * 100
    val_metrics['unmasked_acc'] = (total_unmasked_correct / total_samples) * 100
    
    # Average the other metrics
    for key in val_metrics:
        if key != 'validation_time' and key != 'masked_acc' and key != 'unmasked_acc':
            val_metrics[key] /= total_batches
            
    return val_metrics

def visualize_results(epoch, train_dir, val_dir, model, train_loader, val_loader, device, classes, num_samples=8):
    model.eval()
    
    # Visualize training samples
    visualize_dataset_samples(train_loader, epoch, num_samples, "train", train_dir, model, device, classes)
    
    # Visualize validation samples
    visualize_dataset_samples(val_loader, epoch, num_samples, "val", val_dir, model, device, classes)

def visualize_dataset_samples(data_loader, epoch, num_samples, dataset_type, save_dir, model, device, classes):
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
        binary_mask = outputs['binary_mask']
        masked_images = outputs['masked_input']
        
        # Get predictions
        _, masked_preds = torch.max(outputs['masked_logits'], 1)
        _, unmasked_preds = torch.max(outputs['unmasked_logits'], 1)
        
        # Move tensors to CPU for visualization
        images = images.cpu()
        soft_mask = soft_mask.cpu()
        radial_mask = radial_mask.cpu()
        binary_mask = binary_mask.cpu()
        masked_images = masked_images.cpu()
        labels = labels.cpu()
        masked_preds = masked_preds.cpu()
        unmasked_preds = unmasked_preds.cpu()
        
        # Ensure labels and predictions are 1D for indexing
        if labels.dim() > 1:
            labels = labels.squeeze()
        if masked_preds.dim() > 1:
            masked_preds = masked_preds.squeeze()
        if unmasked_preds.dim() > 1:
            unmasked_preds = unmasked_preds.squeeze()
        
        # Create a figure with rows of images
        fig, axs = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        
        for i in range(num_samples):
            # Original image with unmasked prediction
            img = images[i].numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # Unnormalize
            axs[i, 0].imshow(img)
            # Convert tensor indices to Python ints
            label_idx = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])
            unpred_idx = unmasked_preds[i].item() if isinstance(unmasked_preds[i], torch.Tensor) else int(unmasked_preds[i])
            correct = "✓" if unpred_idx == label_idx else "✗"
            # Use string keys for classes dict
            label_name = classes.get(str(label_idx), str(label_idx))
            pred_name = classes.get(str(unpred_idx), str(unpred_idx))
            axs[i, 0].set_title(f'Original: {label_name}\nPred: {pred_name} {correct}')
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
            
            # Binary Mask
            binary_mask_mean = binary_mask[i].mean().item()
            binary_pct = binary_mask_mean * 100
            axs[i, 3].imshow(binary_mask[i].squeeze(), cmap='binary')
            axs[i, 3].set_title(f'Binary Mask (> {model.upper_mask_level_threshold})\nMasked: {binary_pct:.1f}%')
            axs[i, 3].axis('off')
            
            # Masked image with prediction
            masked_img = masked_images[i].numpy().transpose((1, 2, 0))
            masked_img = masked_img * 0.5 + 0.5  # Unnormalize
            axs[i, 4].imshow(masked_img)
            # Convert tensor indices to Python ints
            mpred_idx = masked_preds[i].item() if isinstance(masked_preds[i], torch.Tensor) else int(masked_preds[i])
            correct_m = "✓" if mpred_idx == label_idx else "✗"
            pred_name_m = classes.get(str(mpred_idx), str(mpred_idx))
            axs[i, 4].set_title(f'Masked\nPred: {pred_name_m} {correct_m}')
            axs[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"results_samples.png"), dpi=150)
        plt.close(fig)
        
        # Save mask grids
        soft_mask_grid = torchvision.utils.make_grid(soft_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(soft_mask_grid, os.path.join(save_dir, f"soft_masks_grid.png"))
        
        radial_mask_grid = torchvision.utils.make_grid(radial_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(radial_mask_grid, os.path.join(save_dir, f"radial_masks_grid.png"))
        
        binary_mask_grid = torchvision.utils.make_grid(binary_mask.repeat(1, 3, 1, 1), nrow=4, normalize=True)
        torchvision.utils.save_image(binary_mask_grid, os.path.join(save_dir, f"binary_masks_grid.png"))
        
        # Save masked images grid
        masked_grid = torchvision.utils.make_grid(masked_images, nrow=4, normalize=True)
        torchvision.utils.save_image(masked_grid, os.path.join(save_dir, f"masked_images_grid.png"))

def plot_training_metrics(log_dir):
    # Load metrics from CSV
    metrics_df = pd.read_csv(os.path.join(log_dir, "training_metrics.csv"))
    
    # Create plots
    plt.figure(figsize=(15, 35))
    
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
    plt.plot(metrics_df['epoch'], metrics_df['train_dynamic_weight'], 'r-', label='Train Dynamic Weight')
    plt.plot(metrics_df['epoch'], metrics_df['val_dynamic_weight'], 'r--', label='Val Dynamic Weight')
    plt.title('Dynamic Masked Loss Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss gap vs Dynamic weight
    plt.subplot(7, 2, 3)
    metrics_df['train_loss_gap'] = metrics_df['train_masked_loss'] - metrics_df['train_unmasked_loss']
    plt.scatter(metrics_df['train_loss_gap'], metrics_df['train_dynamic_weight'], c=metrics_df['epoch'], cmap='viridis')
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
    
    # Accuracies plot
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
    plt.plot(metrics_df['train_dynamic_weight'], metrics_df['train_masked_acc'], 'bo-', label='Train Masked Acc')
    plt.plot(metrics_df['val_dynamic_weight'], metrics_df['val_masked_acc'], 'go-', label='Val Masked Acc')
    for i, txt in enumerate(metrics_df['epoch']):
        plt.annotate(txt, (metrics_df['train_dynamic_weight'].iloc[i], metrics_df['train_masked_acc'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center')
    plt.title('Dynamic Weight vs Masked Accuracy')
    plt.xlabel('Dynamic Weight')
    plt.ylabel('Masked Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Soft Masking percentage plot
    plt.subplot(7, 2, 8)
    plt.plot(metrics_df['epoch'], metrics_df['train_masked_pixels_pct'], 'm-', label='Train Soft Masked %')
    plt.plot(metrics_df['epoch'], metrics_df['val_masked_pixels_pct'], 'm--', label='Val Soft Masked %')
    plt.title('Soft Masking Percentage')
    plt.xlabel('Epoch')
    plt.ylabel('Masked Pixels (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Binary Masking percentage plot
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
    
    # Regularization parameters comparison
    plt.subplot(7, 2, 11)
    plt.plot(metrics_df['epoch'], metrics_df['lambda_mask'], 'r-', label='Lambda Mask')
    plt.plot(metrics_df['epoch'], metrics_df['lambda_fully_masked'], 'c-', label='Lambda Binary')
    plt.plot(metrics_df['epoch'], metrics_df['train_dynamic_weight'], 'g-', label='Dynamic Weight')
    plt.title('Regularization Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Alignment loss plot
    plt.subplot(7, 2, 12)
    if 'train_alignment_loss' in metrics_df.columns and 'val_alignment_loss' in metrics_df.columns:
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
    
    # Alignment divergence plot
    plt.subplot(7, 2, 13)
    if 'train_alignment_divergence' in metrics_df.columns and 'val_alignment_divergence' in metrics_df.columns:
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
    
    # Create a specific plot for dynamic weight vs loss gap
    plt.figure(figsize=(10, 8))
    metrics_df['val_loss_gap'] = metrics_df['val_masked_loss'] - metrics_df['val_unmasked_loss']
    
    plt.scatter(metrics_df['train_loss_gap'], metrics_df['train_dynamic_weight'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, alpha=0.7, label='Train')
    plt.scatter(metrics_df['val_loss_gap'], metrics_df['val_dynamic_weight'], 
                c=metrics_df['epoch'], cmap='viridis', s=100, marker='x', alpha=0.7, label='Validation')
    
    for i, txt in enumerate(metrics_df['epoch']):
        plt.annotate(txt, (metrics_df['train_loss_gap'].iloc[i], metrics_df['train_dynamic_weight'].iloc[i]), 
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
    
    # Create a plot for alignment divergence vs. masked accuracy
    if 'train_alignment_divergence' in metrics_df.columns and 'val_alignment_divergence' in metrics_df.columns:
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

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    
    print("\nEVALUATION ON TEST SET")
    print("=" * 50)
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    # Print improved evaluation summary
    print("\nTest Results:")
    print(f"  Unmasked: Acc={test_metrics['unmasked_acc']:.2f}%, Loss={test_metrics['unmasked_loss']:.4f}")
    print(f"  Masked:   Acc={test_metrics['masked_acc']:.2f}%, Loss={test_metrics['masked_loss']:.4f}")
    print(f"  Masking:  Soft={test_metrics['masked_pixels_pct']:.1f}%, Binary={test_metrics['fully_masked_pct']:.1f}%")
    print(f"  Losses:   Total={test_metrics['total_loss']:.4f}, Mask={test_metrics['masking_loss']:.4f}, Binary={test_metrics['fully_masked_loss']:.4f}")
    print(f"  Dynamic:  Weight={test_metrics['dynamic_weight']:.2f}, Alignment={test_metrics['alignment_loss']:.4f}")
    print("=" * 50)
    
    return test_metrics

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    print(f"\nTraining on device: {device}")
    print(f"Model parameters device: {next(model.parameters()).device}")
    
    train_metrics = {
        'total_loss': 0.0,
        'masked_loss': 0.0,
        'unmasked_loss': 0.0,
        'masking_loss': 0.0,
        'fully_masked_loss': 0.0,
        'alignment_loss': 0.0,
        'alignment_divergence': 0.0,
        'mask_mean': 0.0,
        'binary_mask_mean': 0.0,
        'fully_masked_pct': 0.0,
        'masked_acc': 0.0,
        'unmasked_acc': 0.0,
        'dynamic_weight': 0.0,
        'masked_pixels_pct': 0.0,
        'training_time': 0.0
    }
    
    total_batches = len(train_loader)
    total_samples = 0
    total_masked_correct = 0
    total_unmasked_correct = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        # Ensure labels shape is [batch_size]
        if labels.dim() > 1 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        if batch_idx == 0:  # Print device info for first batch
            print(f"First batch - Images device: {images.device}, Labels device: {labels.device}")
        
        # Forward pass
        outputs = model(images)
        loss_dict = criterion(outputs, labels)
        
        # Calculate accuracies
        _, masked_preds = torch.max(outputs['masked_logits'], 1)
        _, unmasked_preds = torch.max(outputs['unmasked_logits'], 1)
        
        # Count correct predictions
        batch_size = labels.size(0)
        total_samples += batch_size
        total_masked_correct += (masked_preds == labels).sum().item()
        total_unmasked_correct += (unmasked_preds == labels).sum().item()
        
        # Calculate per-batch accuracy for progress display only
        masked_acc = (masked_preds == labels).sum().item() / batch_size * 100
        unmasked_acc = (unmasked_preds == labels).sum().item() / batch_size * 100
        
        # Backward and optimize
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # Update metrics
        train_metrics['total_loss'] += loss_dict['total_loss'].item()
        train_metrics['masked_loss'] += loss_dict['masked_loss'].item()
        train_metrics['unmasked_loss'] += loss_dict['unmasked_loss'].item()
        train_metrics['masking_loss'] += loss_dict['masking_loss'].item()
        train_metrics['fully_masked_loss'] += loss_dict['fully_masked_loss'].item()
        train_metrics['alignment_loss'] += loss_dict['alignment_loss'].item()
        train_metrics['alignment_divergence'] += loss_dict['alignment_divergence'].item()
        train_metrics['mask_mean'] += outputs['soft_mask'].mean().item()
        train_metrics['binary_mask_mean'] += outputs['radial_mask'].mean().item()
        train_metrics['fully_masked_pct'] += loss_dict['fully_masked_pct']
        train_metrics['dynamic_weight'] += loss_dict['dynamic_weight'].item()
        train_metrics['masked_pixels_pct'] += (outputs['soft_mask'].mean().item() * 100)
        
        # Update progress bar with masking metrics
        progress_bar.set_postfix({
            'Loss': f"{loss_dict['total_loss'].item():.4f}",
            'Masked Acc': f"{masked_acc:.2f}%",
            'Unmasked Acc': f"{unmasked_acc:.2f}%",
            'Soft Mask': f"{outputs['soft_mask'].mean().item()*100:.1f}%",
            'Binary Mask': f"{loss_dict['fully_masked_pct']:.1f}%",
            'Dynamic W': f"{loss_dict['dynamic_weight'].item():.2f}"
        })
    
    end_time = time.time()
    train_metrics['training_time'] = end_time - start_time
    
    # Calculate global accuracy
    train_metrics['masked_acc'] = (total_masked_correct / total_samples) * 100
    train_metrics['unmasked_acc'] = (total_unmasked_correct / total_samples) * 100
    
    # Average the other metrics
    for key in train_metrics:
        if key != 'training_time' and key != 'masked_acc' and key != 'unmasked_acc':
            train_metrics[key] /= total_batches
            
    return train_metrics

def log_efficiency_metrics(metrics, log_dir, epoch=None):
    """Log efficiency metrics to file"""
    if epoch is not None:
        filename = os.path.join(log_dir, f"efficiency_epoch_{epoch}.txt")
    else:
        filename = os.path.join(log_dir, "efficiency.txt")
    
    # Get metrics with default values
    training_time = metrics.get('training_time', 0.0)
    validation_time = metrics.get('validation_time', 0.0)
    total_time = metrics.get('total_time', 0.0)
    
    with open(filename, "w") as f:
        f.write("Epoch,Training Time (s),Validation Time (s),Total Epoch Time (s)\n")
        f.write(f"{epoch if epoch is not None else 'Initial'},{training_time:.2f},{validation_time:.2f},{total_time:.2f}\n")
        
        if epoch == "final":
            f.write("\nSUMMARY\n")
            f.write(f"Total experiment time: {metrics.get('total_experiment_time', 0.0):.2f} seconds ({metrics.get('total_experiment_time', 0.0)/60:.2f} minutes)\n")
            f.write(f"Average epoch time: {metrics.get('avg_epoch_time', 0.0):.2f} seconds\n")
            f.write(f"Average validation time: {metrics.get('avg_validation_time', 0.0):.2f} seconds\n")
            f.write(f"Average training time per epoch: {metrics.get('avg_training_time', 0.0):.2f} seconds\n")
            f.write(f"Total epochs: {metrics.get('total_epochs', 0)}\n")
            f.write(f"Visualization interval: Every {metrics.get('visualization_interval', 1)} epochs\n")
            f.write(f"Checkpoint interval: Every {metrics.get('checkpoint_interval', 1)} epochs\n")

def measure_efficiency(model, data_loader, device, num_batches=10):
    """Measure model efficiency metrics"""
    model.eval()
    
    # Initialize metrics
    metrics = {
        'training_time': 0.0,
        'validation_time': 0.0,
        'total_time': 0.0
    }
    
    # Warm up
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 2:  # Warm up for 2 batches
                break
            images = images.to(device)
            _ = model(images)
    
    # Measure metrics
    with torch.no_grad():
        start_time = time.time()
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
        end_time = time.time()
        metrics['validation_time'] = end_time - start_time
        metrics['total_time'] = metrics['validation_time']
    
    return metrics

def setup_experiment(config):
    """Setup experiment directories and save configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['log_dir'], config['experiment_name'], f"{config['experiment_name']}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create visualization directories
    train_viz_dir = os.path.join(log_dir, "visualizations", "train")
    val_viz_dir = os.path.join(log_dir, "visualizations", "val")
    os.makedirs(train_viz_dir, exist_ok=True)
    os.makedirs(val_viz_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    return log_dir, checkpoint_dir, train_viz_dir, val_viz_dir

def load_pretrained_weights(model, pretrained_path, device, strict=True):
    """
    Load pre-trained weights for fine-tuning.
    
    Args:
        model: The model to load weights into
        pretrained_path: Path to the pre-trained weights file
        device: Device to load the weights onto
        strict: Whether to strictly enforce that the keys in state_dict match the model
        
    Returns:
        Dictionary containing any missing or unexpected keys
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pre-trained weights file not found: {pretrained_path}")
    
    print(f"Loading pre-trained weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    return {
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys
    }

def perform_initial_evaluation(model, train_loader, val_loader, test_loader, criterion, device, log_dir, classes):
    """Perform comprehensive initial evaluation before training"""
    print("\nINITIAL EVALUATION (Before Training)")
    print("=====================================")
    
    # Create initial evaluation directory
    initial_eval_dir = os.path.join(log_dir, "initial_evaluation")
    os.makedirs(initial_eval_dir, exist_ok=True)
    
    # Evaluate on training set
    print("\nTraining Set:")
    train_metrics = evaluate_model(model, train_loader, criterion, device)
    
    # Evaluate on validation set
    print("\nValidation Set:")
    val_metrics = evaluate_model(model, val_loader, criterion, device)
    
    # Evaluate on test set
    print("\nTest Set:")
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Save initial metrics to CSV
    initial_metrics_file = os.path.join(initial_eval_dir, "initial_metrics.csv")
    with open(initial_metrics_file, "w") as f:
        f.write("dataset,unmasked_acc,unmasked_loss,masked_acc,masked_loss,masking_loss,masked_pixels_pct,fully_masked_pct\n")
        f.write(f"train,{train_metrics['unmasked_acc']:.2f},{train_metrics['unmasked_loss']:.6f},{train_metrics['masked_acc']:.2f},{train_metrics['masked_loss']:.6f},{train_metrics['masking_loss']:.6f},{train_metrics['masked_pixels_pct']:.2f},{train_metrics['fully_masked_pct']:.2f}\n")
        f.write(f"val,{val_metrics['unmasked_acc']:.2f},{val_metrics['unmasked_loss']:.6f},{val_metrics['masked_acc']:.2f},{val_metrics['masked_loss']:.6f},{val_metrics['masking_loss']:.6f},{val_metrics['masked_pixels_pct']:.2f},{val_metrics['fully_masked_pct']:.2f}\n")
        f.write(f"test,{test_metrics['unmasked_acc']:.2f},{test_metrics['unmasked_loss']:.6f},{test_metrics['masked_acc']:.2f},{test_metrics['masked_loss']:.6f},{test_metrics['masking_loss']:.6f},{test_metrics['masked_pixels_pct']:.2f},{test_metrics['fully_masked_pct']:.2f}\n")
    
    # Visualize initial results
    print("\nGenerating initial visualizations...")
    visualize_results(0, initial_eval_dir, initial_eval_dir, model, train_loader, val_loader, device, classes)
    
    print(f"\nInitial evaluation complete. Results saved to {initial_eval_dir}")
    print("=====================================\n")
    
    return train_metrics, val_metrics, test_metrics

def save_checkpoint(model, checkpoint_path, is_best=False):
    """Save model state dict"""
    torch.save(model.state_dict(), checkpoint_path)
    if is_best:
        print(f"\nBest model saved to {checkpoint_path}")
    else:
        print(f"\nCheckpoint saved to {checkpoint_path}")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config, log_dir, checkpoint_dir, train_viz_dir, val_viz_dir, classes):
    """Main training loop"""
    print("\nStarting training...")
    print("=" * 50)
    
    # Initialize metrics tracking
    metrics_history = {
        'train_masked_acc': [], 'train_unmasked_acc': [],
        'train_masked_loss': [], 'train_unmasked_loss': [],
        'train_total_loss': [], 'train_masking_loss': [],
        'train_fully_masked_loss': [], 'train_dynamic_weight': [],
        'train_alignment_loss': [], 'train_alignment_divergence': [],
        'train_masked_pixels_pct': [], 'train_fully_masked_pct': [],
        'val_masked_acc': [], 'val_unmasked_acc': [],
        'val_masked_loss': [], 'val_unmasked_loss': [],
        'val_total_loss': [], 'val_masking_loss': [],
        'val_fully_masked_loss': [], 'val_dynamic_weight': [],
        'val_alignment_loss': [], 'val_alignment_divergence': [],
        'val_masked_pixels_pct': [], 'val_fully_masked_pct': [],
        'lambda_mask': [], 'lambda_fully_masked': [], 'lambda_alignment': []
    }
    
    # Initialize timing variables
    total_experiment_time = time.time()
    epoch_times = []
    validation_times = []
    
    # Track best validation accuracy
    best_val_acc = 0.0
    best_epoch = 0
    
    # Create epoch tracker file
    epoch_tracker_path = os.path.join(log_dir, 'epoch_tracker.txt')
    with open(epoch_tracker_path, 'w') as f:
        f.write("Epoch tracking:\n")
    
    # Create efficiency log file
    with open(os.path.join(log_dir, "efficiency.txt"), "w") as f:
        f.write("Epoch,Training Time (s),Validation Time (s),Total Epoch Time (s)\n")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Update lambda values based on current epoch
        if epoch >= config['start_epoch_mask']:
            progress_ratio = (epoch - config['start_epoch_mask']) / max(1, (config['num_epochs'] - 1 - config['start_epoch_mask']))
            current_lambda_mask = config['initial_lambda_mask'] + (config['final_lambda_mask'] - config['initial_lambda_mask']) * progress_ratio
        else:
            current_lambda_mask = 0.0
        criterion.lambda_mask = current_lambda_mask
        
        if epoch >= config['start_epoch_fully_masked']:
            progress_ratio = (epoch - config['start_epoch_fully_masked']) / max(1, (config['num_epochs'] - 1 - config['start_epoch_fully_masked']))
            current_lambda_fully_masked = config['initial_lambda_fully_masked'] + (config['final_lambda_fully_masked'] - config['initial_lambda_fully_masked']) * progress_ratio
        else:
            current_lambda_fully_masked = 0.0
        criterion.lambda_fully_masked = current_lambda_fully_masked
        
        if epoch >= config['start_epoch_alignment']:
            progress_ratio = (epoch - config['start_epoch_alignment']) / max(1, (config['num_epochs'] - 1 - config['start_epoch_alignment']))
            current_lambda_alignment = config['initial_lambda_alignment'] + (config['final_lambda_alignment'] - config['initial_lambda_alignment']) * progress_ratio
        else:
            current_lambda_alignment = 0.0
        criterion.lambda_alignment = current_lambda_alignment
        
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Run validation
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Record validation time
        validation_times.append(val_metrics['validation_time'])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log timing information
        with open(os.path.join(log_dir, "efficiency.txt"), "a") as f:
            f.write(f"{epoch+1},{epoch_time - val_metrics['validation_time']:.2f},{val_metrics['validation_time']:.2f},{epoch_time:.2f}\n")
        
        # Update metrics history
        metrics_history['train_masked_acc'].append(train_metrics['masked_acc'])
        metrics_history['train_unmasked_acc'].append(train_metrics['unmasked_acc'])
        metrics_history['train_masked_loss'].append(train_metrics['masked_loss'])
        metrics_history['train_unmasked_loss'].append(train_metrics['unmasked_loss'])
        metrics_history['train_total_loss'].append(train_metrics['total_loss'])
        metrics_history['train_masking_loss'].append(train_metrics['masking_loss'])
        metrics_history['train_fully_masked_loss'].append(train_metrics['fully_masked_loss'])
        metrics_history['train_dynamic_weight'].append(train_metrics['dynamic_weight'])
        metrics_history['train_alignment_loss'].append(train_metrics['alignment_loss'])
        metrics_history['train_alignment_divergence'].append(train_metrics['alignment_divergence'])
        metrics_history['train_masked_pixels_pct'].append(train_metrics['masked_pixels_pct'])
        metrics_history['train_fully_masked_pct'].append(train_metrics['fully_masked_pct'])
        
        metrics_history['val_masked_acc'].append(val_metrics['masked_acc'])
        metrics_history['val_unmasked_acc'].append(val_metrics['unmasked_acc'])
        metrics_history['val_masked_loss'].append(val_metrics['masked_loss'])
        metrics_history['val_unmasked_loss'].append(val_metrics['unmasked_loss'])
        metrics_history['val_total_loss'].append(val_metrics['total_loss'])
        metrics_history['val_masking_loss'].append(val_metrics['masking_loss'])
        metrics_history['val_fully_masked_loss'].append(val_metrics['fully_masked_loss'])
        metrics_history['val_dynamic_weight'].append(val_metrics['dynamic_weight'])
        metrics_history['val_alignment_loss'].append(val_metrics['alignment_loss'])
        metrics_history['val_alignment_divergence'].append(val_metrics['alignment_divergence'])
        metrics_history['val_masked_pixels_pct'].append(val_metrics['masked_pixels_pct'])
        metrics_history['val_fully_masked_pct'].append(val_metrics['fully_masked_pct'])
        
        metrics_history['lambda_mask'].append(current_lambda_mask)
        metrics_history['lambda_fully_masked'].append(current_lambda_fully_masked)
        metrics_history['lambda_alignment'].append(current_lambda_alignment)
        
        # Print consolidated metrics
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("--------------------------------------------------")
        print("Training Metrics:")
        print(f"  Unmasked: Acc={train_metrics['unmasked_acc']:.2f}%, Loss={train_metrics['unmasked_loss']:.4f}")
        print(f"  Masked:   Acc={train_metrics['masked_acc']:.2f}%, Loss={train_metrics['masked_loss']:.4f}")
        print(f"  Masking:  Soft={train_metrics['masked_pixels_pct']:.1f}%, Binary={train_metrics['fully_masked_pct']:.1f}%")
        print(f"  Losses:   Total={train_metrics['total_loss']:.4f}, Mask={train_metrics['masking_loss']:.4f}, Binary={train_metrics['fully_masked_loss']:.4f}")
        print(f"  Dynamic:  Weight={train_metrics['dynamic_weight']:.2f}")
        print(f"  Alignment: Divergence={train_metrics['alignment_divergence']:.4f}, Loss={train_metrics['alignment_loss']:.4f}")
        print("--------------------------------------------------")
        print("Validation Metrics:")
        print(f"  Unmasked: Acc={val_metrics['unmasked_acc']:.2f}%, Loss={val_metrics['unmasked_loss']:.4f}")
        print(f"  Masked:   Acc={val_metrics['masked_acc']:.2f}%, Loss={val_metrics['masked_loss']:.4f}")
        print(f"  Masking:  Soft={val_metrics['masked_pixels_pct']:.1f}%, Binary={val_metrics['fully_masked_pct']:.1f}%")
        print(f"  Losses:   Total={val_metrics['total_loss']:.4f}, Mask={val_metrics['masking_loss']:.4f}, Binary={val_metrics['fully_masked_loss']:.4f}")
        print(f"  Dynamic:  Weight={val_metrics['dynamic_weight']:.2f}")
        print(f"  Alignment: Divergence={val_metrics['alignment_divergence']:.4f}, Loss={val_metrics['alignment_loss']:.4f}")
        print("--------------------------------------------------")
        
        # Save checkpoint after each epoch
        final_checkpoint_path = os.path.join(checkpoint_dir, 'final_epoch.pth')
        save_checkpoint(model, final_checkpoint_path, is_best=False)
        
        # Update best model if validation loss improved
        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch + 1
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_epoch.pth')
            save_checkpoint(model, best_checkpoint_path, is_best=True)
        
        # Update epoch tracker
        with open(epoch_tracker_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}: Final checkpoint saved\n")
            if epoch + 1 == best_epoch:
                f.write(f"Epoch {epoch + 1}: Best checkpoint saved (val_acc: {val_metrics['masked_acc']:.2f}%)\n")
        
        # Visualize results if needed
        if (epoch + 1) % config['visualization_interval'] == 0:
            # Create epoch-specific visualization directories
            epoch_train_viz_dir = os.path.join(train_viz_dir, f"epoch_{epoch+1}")
            epoch_val_viz_dir = os.path.join(val_viz_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_train_viz_dir, exist_ok=True)
            os.makedirs(epoch_val_viz_dir, exist_ok=True)
            
            visualize_results(epoch, epoch_train_viz_dir, epoch_val_viz_dir, model, train_loader, val_loader, device, classes)
    
    # Calculate average times
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_validation_time = sum(validation_times) / len(validation_times)
    
    # Save final metrics history
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df['epoch'] = range(1, len(metrics_df) + 1)  # Add epoch column starting from 1
    metrics_df.to_csv(os.path.join(log_dir, "training_metrics.csv"), index=False)
    
    # Log final efficiency summary
    final_efficiency = {
        'total_experiment_time': total_experiment_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_validation_time': avg_validation_time,
        'avg_training_time': avg_epoch_time - avg_validation_time,
        'total_epochs': config['num_epochs'],
        'visualization_interval': config['visualization_interval'],
        'checkpoint_interval': config['checkpoint_interval']
    }
    log_efficiency_metrics(final_efficiency, log_dir, "final")
    
    return metrics_history 