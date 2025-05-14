import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy.stats as stats
from captum.attr import IntegratedGradients
from matplotlib.colors import LinearSegmentedColormap
from toolkits import apply_radial_mask
import seaborn as sns

class SHAPVisualizer:
    def __init__(self, pretrained_model, interpretable_model, data_loader, device, num_samples=10000, num_visualize=10):
        """
        Initialize visualizer.
        
        Args:
            pretrained_model: The pretrained model for baseline attributions
            interpretable_model: The interpretable model for masked/unmasked paths
            data_loader: DataLoader for the dataset
            device: Device to run computations on
            num_samples: Number of samples to use for attribution calculation (default: 10000 samples)
            num_visualize: Number of samples to visualize (default: 10 samples)
        """
        self.pretrained_model = pretrained_model
        self.interpretable_model = interpretable_model
        self.data_loader = data_loader
        self.device = device
        self.num_samples = min(num_samples, len(data_loader.dataset))
        self.num_visualize = num_visualize
        
        print(f"\nInitializing visualizer with {self.num_samples} samples for analysis and {self.num_visualize} samples for visualization")
        
        # Create background dataset for IG
        self.background = self._create_background()
        
    def _create_background(self):
        """Create background dataset for IG calculation"""
        print("Creating background dataset...")
        # Sample images from the dataset
        all_images = []
        all_labels = []
        
        # Get all samples from the dataset
        for images, labels in tqdm(self.data_loader, desc="Loading dataset"):
            all_images.append(images)
            all_labels.append(labels)
            if len(all_images) * images.size(0) >= self.num_samples:
                break
        
        # Take only the number of samples we need
        background_images = torch.cat(all_images, dim=0)[:self.num_samples]
        background_labels = torch.cat(all_labels, dim=0)[:self.num_samples]
        
        print(f"Selected {len(background_images)} samples for IG analysis")
        print(f"Image shape: {background_images[0].shape}, Total features per image: {np.prod(background_images[0].shape)}")
        print(f"Total number of features across all samples: {len(background_images) * np.prod(background_images[0].shape):,}")
        return background_images, background_labels
    
    def _model_predict(self, x):
        """Wrapper for pretrained model prediction for IG - baseline path"""
        # Use pretrained model directly
        logits = self.pretrained_model(x)
        return logits
    
    def _model_predict_masked(self, x):
        """Wrapper for masked model prediction for IG - direct attribution to masked input"""
        # Get the masked input directly from the model's forward pass
        outputs = self.interpretable_model(x)
        masked_x = outputs['masked_x']
        # Get predictions using the classifier on masked input
        logits = self.interpretable_model.classifier(masked_x)
        return logits
    
    def _model_predict_unmasked(self, x):
        """Wrapper for unmasked model prediction for IG"""
        # For unmasked path, just use the classifier directly
        logits = self.interpretable_model.classifier(x)
        return logits
    
    def calculate_attributions(self):
        """Calculate Integrated Gradients attributions for all paths in batches"""
        print("\nCalculating Integrated Gradients attributions...")
        
        # Initialize storage for all attributions
        all_attributions = {
            'baseline': [],
            'unmasked': [],
            'masked': []
        }
        
        # Process in batches
        batch_size = 10  # Process 10 samples at a time
        num_batches = (self.num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches", total=num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, self.num_samples)
            
            # Get batch of images and labels
            batch_images = self.background[0][start_idx:end_idx].to(self.device)
            batch_labels = self.background[1][start_idx:end_idx].to(self.device)
            
            # Create baseline (black image)
            baseline = torch.zeros_like(batch_images).to(self.device)
            
            # Initialize IG explainers
            baseline_explainer = IntegratedGradients(self._model_predict)
            unmasked_explainer = IntegratedGradients(self._model_predict_unmasked)
            masked_explainer = IntegratedGradients(self._model_predict_masked)
            
            # Calculate attributions for this batch
            with torch.no_grad():
                # Baseline path
                attributions_baseline = baseline_explainer.attribute(
                    batch_images, 
                    baseline,
                    target=batch_labels,
                    n_steps=50
                )
                
                # Unmasked path
                attributions_unmasked = unmasked_explainer.attribute(
                    batch_images,
                    baseline,
                    target=batch_labels,
                    n_steps=50
                )
                
                # Masked path
                attributions_masked = masked_explainer.attribute(
                    batch_images,
                    baseline,
                    target=batch_labels,
                    n_steps=50
                )
            
            # Store batch results
            all_attributions['baseline'].append(attributions_baseline.cpu())
            all_attributions['unmasked'].append(attributions_unmasked.cpu())
            all_attributions['masked'].append(attributions_masked.cpu())
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {end_idx}/{self.num_samples} samples")
        
        # Concatenate all batches
        return {
            'baseline': torch.cat(all_attributions['baseline'], dim=0),
            'unmasked': torch.cat(all_attributions['unmasked'], dim=0),
            'masked': torch.cat(all_attributions['masked'], dim=0)
        }
    
    def analyze_shap_comparison(self, attributions, save_dir, normalize_func=None):
        """Analyze and compare SHAP values between baseline, unmasked, and masked paths"""
        print("\nAnalyzing SHAP value distributions...")
        
        # Create analysis directory
        analysis_dir = os.path.join(save_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Get actual number of samples from the attributions
        num_samples = attributions['baseline'].size(0)
        print(f"Analyzing {num_samples} samples...")
        
        # Calculate statistics for each sample
        baseline_entropy = []
        unmasked_entropy = []
        masked_entropy = []
        baseline_std = []
        unmasked_std = []
        masked_std = []
        baseline_std_norm = []
        unmasked_std_norm = []
        masked_std_norm = []
        
        for i in tqdm(range(num_samples), desc="Analyzing samples"):
            # Get SHAP values for the true class
            true_label = self.background[1][i]
            
            # Use the provided normalization function or default to positive normalization
            if normalize_func is None:
                normalize_func = self._normalize_positive_features
            
            # Get normalized attributions
            normalized_attrs = normalize_func({
                'baseline': attributions['baseline'][i],
                'unmasked': attributions['unmasked'][i],
                'masked': attributions['masked'][i]
            })
            
            # Calculate entropy of normalized SHAP values
            baseline_entropy.append(stats.entropy(normalized_attrs['baseline']))
            unmasked_entropy.append(stats.entropy(normalized_attrs['unmasked']))
            masked_entropy.append(stats.entropy(normalized_attrs['masked']))
            
            # Calculate standard deviation (both normalized and unnormalized)
            baseline_std.append(np.std(attributions['baseline'][i].numpy()))
            unmasked_std.append(np.std(attributions['unmasked'][i].numpy()))
            masked_std.append(np.std(attributions['masked'][i].numpy()))
            
            baseline_std_norm.append(np.std(normalized_attrs['baseline']))
            unmasked_std_norm.append(np.std(normalized_attrs['unmasked']))
            masked_std_norm.append(np.std(normalized_attrs['masked']))
        
        # Plot 1: Entropy Comparison (with normalized values)
        plt.figure(figsize=(15, 5))
        
        # Entropy comparison
        plt.subplot(1, 2, 1)
        plt.boxplot([baseline_entropy, unmasked_entropy, masked_entropy], 
                   labels=['Baseline', 'Unmasked', 'Masked'])
        plt.title('SHAP Value Entropy Comparison (Normalized)')
        plt.ylabel('Entropy')
        
        # Plot entropy differences
        plt.subplot(1, 2, 2)
        unmasked_diff = np.array(unmasked_entropy) - np.array(baseline_entropy)
        masked_diff = np.array(masked_entropy) - np.array(baseline_entropy)
        plt.boxplot([unmasked_diff, masked_diff],
                   labels=['Unmasked - Baseline', 'Masked - Baseline'])
        plt.title('Entropy Difference from Baseline (Normalized)')
        plt.ylabel('Entropy Difference')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'shap_entropy_comparison.png'))
        plt.close()
        
        # Plot 2: Standard Deviation Comparison (both normalized and unnormalized)
        plt.figure(figsize=(15, 5))
        
        # Standard deviation comparison (unnormalized)
        plt.subplot(1, 3, 1)
        plt.boxplot([baseline_std, unmasked_std, masked_std], 
                   labels=['Baseline', 'Unmasked', 'Masked'])
        plt.title('SHAP Value Standard Deviation (Unnormalized)')
        plt.ylabel('Standard Deviation')
        
        # Standard deviation comparison (normalized)
        plt.subplot(1, 3, 2)
        plt.boxplot([baseline_std_norm, unmasked_std_norm, masked_std_norm], 
                   labels=['Baseline', 'Unmasked', 'Masked'])
        plt.title('SHAP Value Standard Deviation (Normalized)')
        plt.ylabel('Standard Deviation')
        
        # Plot standard deviation differences (normalized)
        plt.subplot(1, 3, 3)
        unmasked_diff = np.array(unmasked_std_norm) - np.array(baseline_std_norm)
        masked_diff = np.array(masked_std_norm) - np.array(baseline_std_norm)
        plt.boxplot([unmasked_diff, masked_diff],
                   labels=['Unmasked - Baseline', 'Masked - Baseline'])
        plt.title('Standard Deviation Difference from Baseline (Normalized)')
        plt.ylabel('Standard Deviation Difference')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'shap_std_comparison.png'))
        plt.close()
        
        # Calculate and save statistics
        stats_file = os.path.join(analysis_dir, 'shap_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("SHAP Value Analysis\n")
            f.write("==================\n\n")
            
            f.write("Entropy Statistics (Normalized):\n")
            f.write(f"Baseline Mean: {np.mean(baseline_entropy):.4f}\n")
            f.write(f"Baseline Std: {np.std(baseline_entropy):.4f}\n")
            f.write(f"Unmasked Mean: {np.mean(unmasked_entropy):.4f}\n")
            f.write(f"Unmasked Std: {np.std(unmasked_entropy):.4f}\n")
            f.write(f"Masked Mean: {np.mean(masked_entropy):.4f}\n")
            f.write(f"Masked Std: {np.std(masked_entropy):.4f}\n")
            f.write(f"Entropy Difference (Unmasked - Baseline): {np.mean(unmasked_entropy) - np.mean(baseline_entropy):.4f}\n")
            f.write(f"Entropy Difference (Masked - Baseline): {np.mean(masked_entropy) - np.mean(baseline_entropy):.4f}\n\n")
            
            f.write("Standard Deviation Statistics (Unnormalized):\n")
            f.write(f"Baseline Mean: {np.mean(baseline_std):.4f}\n")
            f.write(f"Baseline Std: {np.std(baseline_std):.4f}\n")
            f.write(f"Unmasked Mean: {np.mean(unmasked_std):.4f}\n")
            f.write(f"Unmasked Std: {np.std(unmasked_std):.4f}\n")
            f.write(f"Masked Mean: {np.mean(masked_std):.4f}\n")
            f.write(f"Masked Std: {np.std(masked_std):.4f}\n\n")
            
            f.write("Standard Deviation Statistics (Normalized):\n")
            f.write(f"Baseline Mean: {np.mean(baseline_std_norm):.4f}\n")
            f.write(f"Baseline Std: {np.std(baseline_std_norm):.4f}\n")
            f.write(f"Unmasked Mean: {np.mean(unmasked_std_norm):.4f}\n")
            f.write(f"Unmasked Std: {np.std(unmasked_std_norm):.4f}\n")
            f.write(f"Masked Mean: {np.mean(masked_std_norm):.4f}\n")
            f.write(f"Masked Std: {np.std(masked_std_norm):.4f}\n")
            f.write(f"Std Difference (Unmasked - Baseline): {np.mean(unmasked_std_norm) - np.mean(baseline_std_norm):.4f}\n")
            f.write(f"Std Difference (Masked - Baseline): {np.mean(masked_std_norm) - np.mean(baseline_std_norm):.4f}\n")
        
        print(f"Analysis saved to {analysis_dir}")
    
    def visualize(self, attributions, save_dir):
        """Visualize attributions and masks for selected samples"""
        print("\nVisualizing results...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Select samples to visualize (only first num_visualize samples)
        background_images = self.background[0][:self.num_visualize].to(self.device)
        background_labels = self.background[1][:self.num_visualize]
        
        # Get model outputs for selected samples
        with torch.no_grad():
            outputs = self.interpretable_model(background_images)
            soft_masks = outputs['soft_mask']
            radial_masks = outputs['radial_mask']
            binary_masks = outputs['binary_mask']
            masked_images = outputs['masked_input']
        
        # Create custom colormap
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue (negative) -> White (neutral) -> Red (positive)
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
        
        # Visualize each sample
        for i in tqdm(range(self.num_visualize), desc="Generating visualizations"):
            # Create directory for this sample
            sample_dir = os.path.join(save_dir, f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # 1. Original visualization (raw attributions)
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 5)
            
            # Original image
            img = background_images[i].cpu().numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # Unnormalize
            
            # Row 1: Original image and raw attributions
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Raw attributions for baseline
            ax2 = fig.add_subplot(gs[0, 1])
            attr_img = attributions['baseline'][i].cpu().numpy().sum(0)
            vmax = max(abs(attr_img.min()), abs(attr_img.max()))
            im2 = ax2.imshow(attr_img, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax2.set_title('Baseline IG (Raw)')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label='Attribution Value')
            
            # Raw attributions for unmasked
            ax3 = fig.add_subplot(gs[0, 2])
            attr_img = attributions['unmasked'][i].cpu().numpy().sum(0)
            im3 = ax3.imshow(attr_img, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax3.set_title('Unmasked IG (Raw)')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, label='Attribution Value')
            
            # Raw attributions for masked
            ax4 = fig.add_subplot(gs[0, 3])
            attr_img = attributions['masked'][i].cpu().numpy().sum(0)
            im4 = ax4.imshow(attr_img, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax4.set_title('Masked IG (Raw)')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, label='Attribution Value')
            
            # Add predictions
            ax5 = fig.add_subplot(gs[0, 4])
            with torch.no_grad():
                baseline_pred = torch.argmax(self.pretrained_model(background_images[i:i+1])).item()
                unmasked_pred = torch.argmax(outputs['unmasked_logits'][i:i+1]).item()
                masked_pred = torch.argmax(outputs['masked_logits'][i:i+1]).item()
            
            ax5.text(0.5, 0.5, 
                    f'Baseline: {baseline_pred}\n'
                    f'Unmasked: {unmasked_pred}\n'
                    f'Masked: {masked_pred}\n'
                    f'True: {background_labels[i]}',
                    ha='center', va='center')
            ax5.axis('off')
            
            # Row 2: Masked image and masks
            ax6 = fig.add_subplot(gs[1, 0])
            masked_img = masked_images[i].cpu().numpy().transpose((1, 2, 0))
            masked_img = masked_img * 0.5 + 0.5  # Unnormalize
            ax6.imshow(masked_img)
            ax6.set_title('Masked Image')
            ax6.axis('off')
            
            # Soft mask
            ax7 = fig.add_subplot(gs[1, 1])
            soft_mask = soft_masks[i].cpu().squeeze()
            im7 = ax7.imshow(soft_mask, cmap='viridis')
            ax7.set_title('Soft Mask')
            ax7.axis('off')
            plt.colorbar(im7, ax=ax7, label='Mask Value')
            
            # Radial mask
            ax8 = fig.add_subplot(gs[1, 2])
            radial_mask = radial_masks[i].cpu().squeeze()
            im8 = ax8.imshow(radial_mask, cmap='viridis')
            ax8.set_title('Radial Mask')
            ax8.axis('off')
            plt.colorbar(im8, ax=ax8, label='Mask Value')
            
            # Binary mask
            ax9 = fig.add_subplot(gs[1, 3])
            binary_mask = binary_masks[i].cpu().squeeze()
            im9 = ax9.imshow(binary_mask, cmap='binary')
            ax9.set_title('Binary Mask')
            ax9.axis('off')
            plt.colorbar(im9, ax=ax9, label='Mask Value')
            
            # Add mask statistics
            ax10 = fig.add_subplot(gs[1, 4])
            ax10.text(0.5, 0.5, 
                     f'Soft Mask Mean: {soft_mask.mean():.3f}\n'
                     f'Radial Mask Mean: {radial_mask.mean():.3f}\n'
                     f'Binary Mask Mean: {binary_mask.mean():.3f}',
                     ha='center', va='center')
            ax10.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'raw_attributions.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create separate figure for attribution distributions
            fig_dist = plt.figure(figsize=(15, 5))
            
            # Get normalized attributions for this sample
            baseline_attr = np.clip(attributions['baseline'][i].cpu().numpy(), 0, None).flatten()
            unmasked_attr = np.clip(attributions['unmasked'][i].cpu().numpy(), 0, None).flatten()
            
            # Normalize to sum to 1
            baseline_attr = baseline_attr / (np.sum(baseline_attr) + 1e-10)
            unmasked_attr = unmasked_attr / (np.sum(unmasked_attr) + 1e-10)
            
            # Plot baseline distribution
            ax1 = fig_dist.add_subplot(1, 2, 1)
            ax1.hist(baseline_attr, bins=50, color='blue', alpha=0.7)
            ax1.axvline(np.mean(baseline_attr), color='red', linestyle='--', alpha=0.5,
                       label=f'Mean: {np.mean(baseline_attr):.3f}')
            ax1.set_title(f'Baseline IG Attribution Distribution\nSample {i}')
            ax1.set_xlabel('Normalized IG Attribution')
            ax1.set_ylabel('Count')
            ax1.legend()
            ax1.grid(True)
            
            # Plot unmasked distribution
            ax2 = fig_dist.add_subplot(1, 2, 2)
            ax2.hist(unmasked_attr, bins=50, color='green', alpha=0.7)
            ax2.axvline(np.mean(unmasked_attr), color='red', linestyle='--', alpha=0.5,
                       label=f'Mean: {np.mean(unmasked_attr):.3f}')
            ax2.set_title(f'Unmasked IG Attribution Distribution\nSample {i}')
            ax2.set_xlabel('Normalized IG Attribution')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'attribution_distributions.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Normalized positive attributions visualization
            fig_norm = plt.figure(figsize=(20, 8))
            gs_norm = fig_norm.add_gridspec(2, 4)
            
            # Original image
            ax1 = fig_norm.add_subplot(gs_norm[0, 0])
            ax1.imshow(img)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Normalized positive attributions for baseline
            ax2 = fig_norm.add_subplot(gs_norm[0, 1])
            attr_img = np.clip(attributions['baseline'][i].cpu().numpy().sum(0), 0, None)
            attr_img = attr_img / (np.sum(attr_img) + 1e-10)
            im2 = ax2.imshow(attr_img, cmap='hot')
            ax2.set_title('Baseline IG (Normalized Positive)')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label='Normalized Attribution')
            
            # Normalized positive attributions for unmasked
            ax3 = fig_norm.add_subplot(gs_norm[0, 2])
            attr_img = np.clip(attributions['unmasked'][i].cpu().numpy().sum(0), 0, None)
            attr_img = attr_img / (np.sum(attr_img) + 1e-10)
            im3 = ax3.imshow(attr_img, cmap='hot')
            ax3.set_title('Unmasked IG (Normalized Positive)')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, label='Normalized Attribution')
            
            # Normalized positive attributions for masked
            ax4 = fig_norm.add_subplot(gs_norm[0, 3])
            attr_img = np.clip(attributions['masked'][i].cpu().numpy().sum(0), 0, None)
            attr_img = attr_img / (np.sum(attr_img) + 1e-10)
            im4 = ax4.imshow(attr_img, cmap='hot')
            ax4.set_title('Masked IG (Normalized Positive)')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, label='Normalized Attribution')
            
            # Add color legend
            ax_legend = fig_norm.add_subplot(gs_norm[1, :])
            ax_legend.axis('off')
            ax_legend.text(0.5, 0.5, 
                         'Color Mapping:\n'
                         'Blue -> White -> Red: Raw attribution values (negative to positive)\n'
                         'Hot colormap: Normalized positive attributions (0 to 1)\n'
                         'Viridis: Mask values (0 to 1)\n'
                         'Binary: Binary mask values (0 or 1)',
                         ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'normalized_attributions.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # 3. Absolute-normalized attributions visualization
            fig_abs = plt.figure(figsize=(20, 8))
            gs_abs = fig_abs.add_gridspec(2, 4)
            
            # Original image
            ax1 = fig_abs.add_subplot(gs_abs[0, 0])
            ax1.imshow(img)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Absolute-normalized attributions for baseline
            ax2 = fig_abs.add_subplot(gs_abs[0, 1])
            attr_img = np.abs(attributions['baseline'][i].cpu().numpy().sum(0))
            attr_img = attr_img / (np.sum(attr_img) + 1e-10)
            im2 = ax2.imshow(attr_img, cmap='hot')
            ax2.set_title('Baseline IG (Absolute Normalized)')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label='Normalized Attribution')
            
            # Absolute-normalized attributions for unmasked
            ax3 = fig_abs.add_subplot(gs_abs[0, 2])
            attr_img = np.abs(attributions['unmasked'][i].cpu().numpy().sum(0))
            attr_img = attr_img / (np.sum(attr_img) + 1e-10)
            im3 = ax3.imshow(attr_img, cmap='hot')
            ax3.set_title('Unmasked IG (Absolute Normalized)')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, label='Normalized Attribution')
            
            # Absolute-normalized attributions for masked
            ax4 = fig_abs.add_subplot(gs_abs[0, 3])
            attr_img = np.abs(attributions['masked'][i].cpu().numpy().sum(0))
            attr_img = attr_img / (np.sum(attr_img) + 1e-10)
            im4 = ax4.imshow(attr_img, cmap='hot')
            ax4.set_title('Masked IG (Absolute Normalized)')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, label='Normalized Attribution')
            
            # Add color legend
            ax_legend = fig_abs.add_subplot(gs_abs[1, :])
            ax_legend.axis('off')
            ax_legend.text(0.5, 0.5, 
                         'Color Mapping:\n'
                         'Hot colormap: Absolute-normalized attributions (0 to 1)\n'
                         'All attributions are converted to absolute values before normalization',
                         ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'absolute_normalized_attributions.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {save_dir}")
    
    def plot_percentile_importance(self, all_shap_values, title_prefix, save_path, num_samples, normalize_func=None):
        """Plot percentile-based importance analysis and feature range analysis"""
        print(f"Analyzing {num_samples} samples for percentile importance...")
        
        # Use the provided normalization function or default to positive normalization
        if normalize_func is None:
            normalize_func = self._normalize_positive_features
        
        # Convert lists to numpy arrays and reshape to per-sample format
        num_features_per_sample = len(all_shap_values['baseline']) // num_samples
        baseline_values = np.array(all_shap_values['baseline']).reshape(num_samples, -1)
        unmasked_values = np.array(all_shap_values['unmasked']).reshape(num_samples, -1)
        masked_values = np.array(all_shap_values['masked']).reshape(num_samples, -1)
        
        # Create figure with three subplots
        plt.figure(figsize=(20, 5))
        
        # Helper function to find threshold for a given percentile
        def find_percentile_value(values, percentile):
            percentiles = []
            for sample in values:
                percentiles.append(np.percentile(np.sort(sample), percentile))
            return np.max(percentiles)  # Use max so all curves are visible
        
        # 1. Top 15% Importance Features (i.e., 85th percentile and above)
        top_15_anchor = find_percentile_value(baseline_values, 85)
        top_15_anchor = max(top_15_anchor, find_percentile_value(unmasked_values, 85))
        top_15_anchor = max(top_15_anchor, find_percentile_value(masked_values, 85))
        max_value = max(np.max(baseline_values), np.max(unmasked_values), np.max(masked_values))
        
        plt.subplot(1, 3, 1)
        thresholds = np.linspace(top_15_anchor, max_value, 20)
        baseline_percentages = []
        unmasked_percentages = []
        masked_percentages = []
        for threshold in thresholds:
            baseline_pct = np.mean([np.sum(sample > threshold) / len(sample) * 100 for sample in baseline_values])
            unmasked_pct = np.mean([np.sum(sample > threshold) / len(sample) * 100 for sample in unmasked_values])
            masked_pct = np.mean([np.sum(sample > threshold) / len(sample) * 100 for sample in masked_values])
            baseline_percentages.append(baseline_pct)
            unmasked_percentages.append(unmasked_pct)
            masked_percentages.append(masked_pct)
        plt.plot(thresholds, baseline_percentages, 'b-o', label='Baseline', markersize=4)
        plt.plot(thresholds, unmasked_percentages, 'g-o', label='Unmasked', markersize=4)
        plt.plot(thresholds, masked_percentages, 'r-o', label='Masked', markersize=4)
        plt.title(f'{title_prefix} Top 15% Importance Features\n(x-axis: 85th percentile to max)')
        plt.xlabel('Importance Threshold')
        plt.ylabel('Average Percentage of Features\nAcross Samples')
        plt.xlim(top_15_anchor, max_value)
        plt.ylim(0, 15)
        plt.legend()
        plt.grid(True)
        
        # 2. Bottom 70% Importance Features (i.e., up to 70th percentile)
        anchors = [
            find_percentile_value(baseline_values, 70),
            find_percentile_value(unmasked_values, 70),
            find_percentile_value(masked_values, 70)
        ]
        bottom_70_anchor = min(anchors)
        plt.subplot(1, 3, 2)
        thresholds = np.linspace(0, bottom_70_anchor, 20)
        baseline_percentages = []
        unmasked_percentages = []
        masked_percentages = []
        for threshold in thresholds:
            baseline_pct = np.mean([np.sum(sample < threshold) / len(sample) * 100 for sample in baseline_values])
            unmasked_pct = np.mean([np.sum(sample < threshold) / len(sample) * 100 for sample in unmasked_values])
            masked_pct = np.mean([np.sum(sample < threshold) / len(sample) * 100 for sample in masked_values])
            baseline_percentages.append(baseline_pct)
            unmasked_percentages.append(unmasked_pct)
            masked_percentages.append(masked_pct)
        plt.plot(thresholds, baseline_percentages, 'b-o', label='Baseline', markersize=4)
        plt.plot(thresholds, unmasked_percentages, 'g-o', label='Unmasked', markersize=4)
        plt.plot(thresholds, masked_percentages, 'r-o', label='Masked', markersize=4)
        plt.title(f'{title_prefix} Bottom 70% Importance Features\n(x-axis: 0 to min 70th percentile)')
        plt.xlabel('Importance Threshold')
        plt.ylabel('Average Percentage of Features\nAcross Samples')
        plt.xlim(0, bottom_70_anchor)
        plt.ylim(0, 70)
        plt.legend()
        plt.grid(True)
        
        # 3. Feature Range Analysis (removing top and bottom x%)
        plt.subplot(1, 3, 3)
        percentages = np.linspace(0, 50, 20)  # 0% to 50%
        
        # Calculate ranges for each percentage
        baseline_ranges = []
        unmasked_ranges = []
        masked_ranges = []
        
        for p in percentages:
            # For each path
            for values, ranges_list in [
                (baseline_values, baseline_ranges),
                (unmasked_values, unmasked_ranges),
                (masked_values, masked_ranges)
            ]:
                # Calculate number of features to remove from each end
                n_remove = int(values.shape[1] * p / 100)
                
                # For each sample, calculate range after removing top/bottom percentages
                sample_ranges = []
                for sample in values:
                    sorted_values = np.sort(sample)
                    remaining_values = sorted_values[n_remove:-n_remove] if n_remove > 0 else sorted_values
                    if len(remaining_values) > 0:
                        value_range = np.max(remaining_values) - np.min(remaining_values)
                        sample_ranges.append(value_range)
                
                # Average range across samples
                ranges_list.append(np.mean(sample_ranges))
        
        plt.plot(percentages, baseline_ranges, 'b-o', label='Baseline', markersize=4)
        plt.plot(percentages, unmasked_ranges, 'g-o', label='Unmasked', markersize=4)
        plt.plot(percentages, masked_ranges, 'r-o', label='Masked', markersize=4)
        
        plt.title(f'{title_prefix} Feature Range Analysis\n(Range after removing top/bottom x%)')
        plt.xlabel('Percentage Removed from Each End')
        plt.ylabel('Average Range of Remaining Features\nAcross Samples')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_threshold_importance(self, all_shap_values, title_prefix, save_path, num_samples, normalize_func=None):
        """Plot threshold-based importance analysis"""
        print(f"Analyzing {num_samples} samples for threshold importance...")
        
        # Use the provided normalization function or default to positive normalization
        if normalize_func is None:
            normalize_func = self._normalize_positive_features
        
        # Convert lists to numpy arrays and reshape to per-sample format
        num_features_per_sample = len(all_shap_values['baseline']) // num_samples
        baseline_values = np.array(all_shap_values['baseline']).reshape(num_samples, -1)
        unmasked_values = np.array(all_shap_values['unmasked']).reshape(num_samples, -1)
        masked_values = np.array(all_shap_values['masked']).reshape(num_samples, -1)
        
        # Create figure with two subplots
        plt.figure(figsize=(15, 5))
        
        # 1. High Importance Features (> x, where x is 0.004 to 0.020)
        plt.subplot(1, 2, 1)
        thresholds = np.linspace(0.004, 0.020, 20)
        
        # Calculate percentages for each threshold
        baseline_percentages = []
        unmasked_percentages = []
        masked_percentages = []
        
        for threshold in thresholds:
            # For each sample, calculate percentage of features above threshold
            baseline_pct = np.mean([np.sum(sample > threshold) / len(sample) * 100 
                                  for sample in baseline_values])
            unmasked_pct = np.mean([np.sum(sample > threshold) / len(sample) * 100 
                                  for sample in unmasked_values])
            masked_pct = np.mean([np.sum(sample > threshold) / len(sample) * 100 
                                for sample in masked_values])
            
            baseline_percentages.append(baseline_pct)
            unmasked_percentages.append(unmasked_pct)
            masked_percentages.append(masked_pct)
        
        plt.plot(thresholds, baseline_percentages, 'b-o', label='Baseline', markersize=4)
        plt.plot(thresholds, unmasked_percentages, 'g-o', label='Unmasked', markersize=4)
        plt.plot(thresholds, masked_percentages, 'r-o', label='Masked', markersize=4)
        
        plt.title(f'{title_prefix} High Importance Features\n(> threshold, range: 0.004-0.020)')
        plt.xlabel('Importance Threshold')
        plt.ylabel('Average Percentage of Features\nAcross Samples')
        plt.xlim(0.004, 0.020)
        plt.legend()
        plt.grid(True)
        
        # 2. Low Importance Features (< x, where x is 0 to 0.005)
        plt.subplot(1, 2, 2)
        thresholds = np.linspace(0, 0.005, 20)
        
        # Calculate percentages for each threshold
        baseline_percentages = []
        unmasked_percentages = []
        masked_percentages = []
        
        for threshold in thresholds:
            # For each sample, calculate percentage of features below threshold
            baseline_pct = np.mean([np.sum(sample < threshold) / len(sample) * 100 
                                  for sample in baseline_values])
            unmasked_pct = np.mean([np.sum(sample < threshold) / len(sample) * 100 
                                  for sample in unmasked_values])
            masked_pct = np.mean([np.sum(sample < threshold) / len(sample) * 100 
                                for sample in masked_values])
            
            baseline_percentages.append(baseline_pct)
            unmasked_percentages.append(unmasked_pct)
            masked_percentages.append(masked_pct)
        
        plt.plot(thresholds, baseline_percentages, 'b-o', label='Baseline', markersize=4)
        plt.plot(thresholds, unmasked_percentages, 'g-o', label='Unmasked', markersize=4)
        plt.plot(thresholds, masked_percentages, 'r-o', label='Masked', markersize=4)
        
        plt.title(f'{title_prefix} Low Importance Features\n(< threshold, range: 0-0.005)')
        plt.xlabel('Importance Threshold')
        plt.ylabel('Average Percentage of Features\nAcross Samples')
        plt.xlim(0, 0.005)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_feature_distributions(self, all_shap_values, title_prefix, raw_save_path, norm_save_path, num_samples, normalize_func=None):
        """Plot raw and normalized distributions for features"""
        paths = ['baseline', 'unmasked', 'masked']
        colors = ['skyblue', 'orange', 'lightgreen']
        
        print(f"Analyzing {num_samples} samples for feature distributions...")
        
        # Use the provided normalization function or default to positive normalization
        if normalize_func is None:
            normalize_func = self._normalize_positive_features
        
        # Calculate total number of features
        num_features = len(all_shap_values['baseline']) // num_samples
        total_features = num_features * num_samples
        print(f"\nPlotting distributions for {title_prefix}")
        print(f"Features per sample: {num_features:,}")
        print(f"Total samples: {num_samples}")
        print(f"Total features: {total_features:,}")
        
        # Plot raw distributions
        fig_raw, axs_raw = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
        fig_raw.suptitle(f'{title_prefix} SHAP Value Distribution\n(from {num_samples} samples, {num_features:,} features per sample)', fontsize=14)
        
        # First pass to find max count for y-axis scaling
        max_count = 0
        for path in paths:
            values = np.array(all_shap_values[path]).flatten()  # Flatten the array
            # Only consider values within range
            values = values[(values >= 0.00) & (values <= 0.02)]
            hist, _ = np.histogram(values, bins=50, range=(0.00, 0.02))  # Set histogram range
            max_count = max(max_count, np.max(hist))
        
        # Plot with consistent y-axis
        for idx, (path, color) in enumerate(zip(paths, colors)):
            values = np.array(all_shap_values[path]).flatten()  # Flatten the array
            # Only consider values within range
            values = values[(values >= 0.00) & (values <= 0.02)]
            n, bins, patches = axs_raw[idx].hist(values, bins=50, color=color, alpha=0.7, range=(0.00, 0.02))  # Set histogram range
            
            # Add count values on top of each bar
            for i in range(len(n)):
                if n[i] > 0:  # Only add text if there are values in the bin
                    axs_raw[idx].text(bins[i] + (bins[i+1] - bins[i])/2, n[i], 
                                    f'{int(n[i]):,}', 
                                    ha='center', va='bottom',
                                    fontsize=5, color='gray')
            
            axs_raw[idx].set_ylabel(f'{path.capitalize()}\nCount')
            mean_val = np.mean(values)
            axs_raw[idx].axvline(x=mean_val, color='red', linestyle='--', alpha=0.5,
                                label=f'Mean: {mean_val:.3f} (n={len(values):,})')
            axs_raw[idx].legend()
            axs_raw[idx].set_ylim(0, max_count * 1.1)
            axs_raw[idx].set_xlim(0.00, 0.02)  # Set x-axis range
        
        axs_raw[-1].set_xlabel('SHAP Value')
        plt.tight_layout()
        plt.savefig(raw_save_path)
        plt.close()
        
        # Plot normalized distributions
        fig_norm, axs_norm = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
        fig_norm.suptitle(f'Normalized {title_prefix} SHAP Value Distribution\n(Sum of attributions = 1 for each sample)\nTotal samples: {num_samples}, Features per sample: {num_features:,}', fontsize=14)
        
        # First pass to find max count for y-axis scaling
        max_count = 0
        for path in paths:
            values = np.array(all_shap_values[path]).flatten()  # Flatten the array
            # Only consider values within range
            values = values[(values >= 0.00) & (values <= 0.02)]
            hist, _ = np.histogram(values, bins=50, range=(0.00, 0.02))
            max_count = max(max_count, np.max(hist))
        
        # Plot with consistent y-axis
        for idx, (path, color) in enumerate(zip(paths, colors)):
            values = np.array(all_shap_values[path]).flatten()  # Flatten the array
            # Only consider values within range
            values = values[(values >= 0.00) & (values <= 0.02)]
            n, bins, patches = axs_norm[idx].hist(values, bins=50, color=color, alpha=0.7, range=(0.00, 0.02))
            
            # Add count values on top of each bar
            for i in range(len(n)):
                if n[i] > 0:  # Only add text if there are values in the bin
                    axs_norm[idx].text(bins[i] + (bins[i+1] - bins[i])/2, n[i], 
                                     f'{int(n[i]):,}', 
                                     ha='center', va='bottom',
                                     fontsize=5, color='gray')
            
            axs_norm[idx].set_ylabel(f'{path.capitalize()}\nCount')
            mean_val = np.mean(values)
            median_val = np.median(values)
            axs_norm[idx].axvline(x=mean_val, color='red', linestyle='--', alpha=0.5,
                                 label=f'Mean: {mean_val:.2e}\nMedian: {median_val:.2e}\n(n={len(values):,})')
            axs_norm[idx].legend()
            axs_norm[idx].set_ylim(0, max_count * 1.1)
            axs_norm[idx].set_xlim(0.00, 0.02)  # Set x-axis range
        
        axs_norm[-1].set_xlabel('Normalized SHAP Value (Attribution / Total Attribution)')
        plt.tight_layout()
        plt.savefig(norm_save_path)
        plt.close()

    def analyze_feature_importance(self, attributions, save_dir, normalize_func=None):
        """Analyze and visualize important vs unimportant features and input pixels based on attributions"""
        print("\nAnalyzing feature and input importance...")
        
        # Create proper directory structure
        analysis_dir = os.path.join(save_dir, 'analysis')
        feature_importance_dir = os.path.join(save_dir, 'feature_importance')
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(feature_importance_dir, exist_ok=True)
        
        # Get actual number of samples from the attributions
        num_samples = attributions['baseline'].size(0)
        print(f"Analyzing {num_samples} samples...")
        
        # Initialize storage for raw and normalized attributions
        raw_attributions = {
            'baseline': [],
            'unmasked': [],
            'masked': []
        }
        normalized_attributions = {
            'baseline': [],
            'unmasked': [],
            'masked': []
        }
        
        # Use the provided normalization function or default to positive normalization
        if normalize_func is None:
            normalize_func = self._normalize_positive_features
        
        # Process in batches
        batch_size = 10
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            # Process MLP features
            for i in range(start_idx, end_idx):
                # Get raw attributions for MLP features (sum across channels)
                baseline_attr = attributions['baseline'][i].numpy().sum(axis=0).flatten()
                unmasked_attr = attributions['unmasked'][i].numpy().sum(axis=0).flatten()
                masked_attr = attributions['masked'][i].numpy().sum(axis=0).flatten()
                
                # Store raw values
                raw_attributions['baseline'].extend(baseline_attr)
                raw_attributions['unmasked'].extend(unmasked_attr)
                raw_attributions['masked'].extend(masked_attr)
                
                # Normalize values using the provided function
                normalized_attrs = normalize_func({
                    'baseline': attributions['baseline'][i],
                    'unmasked': attributions['unmasked'][i],
                    'masked': attributions['masked'][i]
                })
                
                normalized_attributions['baseline'].extend(normalized_attrs['baseline'])
                normalized_attributions['unmasked'].extend(normalized_attrs['unmasked'])
                normalized_attributions['masked'].extend(normalized_attrs['masked'])
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Plot feature distributions using normalized values
        self.plot_feature_distributions(
            normalized_attributions,
            'MLP Features',
            os.path.join(feature_importance_dir, 'mlp_shap_distributions_normalized.png'),
            os.path.join(feature_importance_dir, 'mlp_shap_distributions_normalized.png'),
            num_samples,
            normalize_func
        )
        
        # Plot percentile-based importance analysis
        self.plot_percentile_importance(
            normalized_attributions,
            'MLP Features',
            os.path.join(feature_importance_dir, 'mlp_percentile_importance.png'),
            num_samples,
            normalize_func
        )
        
        # Plot threshold-based importance analysis
        self.plot_threshold_importance(
            normalized_attributions,
            'MLP Features',
            os.path.join(feature_importance_dir, 'mlp_threshold_importance.png'),
            num_samples,
            normalize_func
        )
        
        # Reset storage for input pixels
        raw_attributions = {
            'baseline': [],
            'unmasked': [],
            'masked': []
        }
        normalized_attributions = {
            'baseline': [],
            'unmasked': [],
            'masked': []
        }
        
        # Process input pixels in batches
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            for i in range(start_idx, end_idx):
                # Get raw attributions for input pixels
                baseline_attr = attributions['baseline'][i].numpy().flatten()
                unmasked_attr = attributions['unmasked'][i].numpy().flatten()
                masked_attr = attributions['masked'][i].numpy().flatten()
                
                # Store raw values
                raw_attributions['baseline'].extend(baseline_attr)
                raw_attributions['unmasked'].extend(unmasked_attr)
                raw_attributions['masked'].extend(masked_attr)
                
                # Normalize values using the provided function
                normalized_attrs = normalize_func({
                    'baseline': attributions['baseline'][i],
                    'unmasked': attributions['unmasked'][i],
                    'masked': attributions['masked'][i]
                })
                
                normalized_attributions['baseline'].extend(normalized_attrs['baseline'])
                normalized_attributions['unmasked'].extend(normalized_attrs['unmasked'])
                normalized_attributions['masked'].extend(normalized_attrs['masked'])
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Plot input pixel distributions using normalized values
        self.plot_feature_distributions(
            normalized_attributions,
            'Input Pixels',
            os.path.join(feature_importance_dir, 'input_shap_distributions_normalized.png'),
            os.path.join(feature_importance_dir, 'input_shap_distributions_normalized.png'),
            num_samples,
            normalize_func
        )
        
        # Plot percentile-based importance analysis for input pixels
        self.plot_percentile_importance(
            normalized_attributions,
            'Input Pixels',
            os.path.join(feature_importance_dir, 'input_percentile_importance.png'),
            num_samples,
            normalize_func
        )
        
        # Plot threshold-based importance analysis for input pixels
        self.plot_threshold_importance(
            normalized_attributions,
            'Input Pixels',
            os.path.join(feature_importance_dir, 'input_threshold_importance.png'),
            num_samples,
            normalize_func
        )

    def _get_correct_predictions(self, images, labels):
        """Get indices of samples where all three paths predict correctly"""
        with torch.no_grad():
            # Move inputs to the correct device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get predictions from all three paths
            baseline_preds = torch.argmax(self.pretrained_model(images), dim=1)
            unmasked_preds = torch.argmax(self.interpretable_model.classifier(images), dim=1)
            masked_preds = torch.argmax(self.interpretable_model(images)['masked_logits'], dim=1)
            
            # Find samples where all paths are correct
            all_correct = (baseline_preds == labels) & (unmasked_preds == labels) & (masked_preds == labels)
            unmasked_baseline_correct = (baseline_preds == labels) & (unmasked_preds == labels)
            
            return {
                'all_correct': all_correct,
                'unmasked_baseline_correct': unmasked_baseline_correct,
                'baseline_correct': (baseline_preds == labels),
                'unmasked_correct': (unmasked_preds == labels),
                'masked_correct': (masked_preds == labels)
            }

    def _save_prediction_stats(self, correct_predictions, save_dir):
        """Save prediction statistics to a text file"""
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        stats = {
            'total_samples': self.num_samples,
            'all_correct': correct_predictions['all_correct'].sum().item(),
            'unmasked_baseline_correct': correct_predictions['unmasked_baseline_correct'].sum().item(),
            'baseline_correct': correct_predictions['baseline_correct'].sum().item(),
            'unmasked_correct': correct_predictions['unmasked_correct'].sum().item(),
            'masked_correct': correct_predictions['masked_correct'].sum().item()
        }
        
        # Calculate percentages
        stats['all_correct_pct'] = (stats['all_correct'] / stats['total_samples']) * 100
        stats['unmasked_baseline_correct_pct'] = (stats['unmasked_baseline_correct'] / stats['total_samples']) * 100
        stats['baseline_correct_pct'] = (stats['baseline_correct'] / stats['total_samples']) * 100
        stats['unmasked_correct_pct'] = (stats['unmasked_correct'] / stats['total_samples']) * 100
        stats['masked_correct_pct'] = (stats['masked_correct'] / stats['total_samples']) * 100
        
        # Save stats
        with open(os.path.join(save_dir, 'prediction_stats.txt'), 'w') as f:
            f.write("Prediction Statistics\n")
            f.write("===================\n\n")
            f.write(f"Total samples analyzed: {stats['total_samples']}\n\n")
            f.write("Correct Predictions:\n")
            f.write(f"All paths correct: {stats['all_correct']} ({stats['all_correct_pct']:.2f}%)\n")
            f.write(f"Unmasked + Baseline correct: {stats['unmasked_baseline_correct']} ({stats['unmasked_baseline_correct_pct']:.2f}%)\n")
            f.write(f"Baseline correct: {stats['baseline_correct']} ({stats['baseline_correct_pct']:.2f}%)\n")
            f.write(f"Unmasked correct: {stats['unmasked_correct']} ({stats['unmasked_correct_pct']:.2f}%)\n")
            f.write(f"Masked correct: {stats['masked_correct']} ({stats['masked_correct_pct']:.2f}%)\n")
        
        return stats

    def _filter_attributions(self, attributions, mask):
        """Filter attributions based on a boolean mask"""
        # Move mask to CPU if it's on GPU
        mask = mask.cpu()
        return {
            'baseline': attributions['baseline'][mask],
            'unmasked': attributions['unmasked'][mask],
            'masked': attributions['masked'][mask]
        }

    def _run_sample_visualization(self, attributions, save_dir):
        """Run visualization for individual samples"""
        print("\nRunning sample visualizations...")
        sample_vis_dir = os.path.join(save_dir, 'sample_vis')
        os.makedirs(sample_vis_dir, exist_ok=True)
        self.visualize(attributions, sample_vis_dir)

    def _normalize_positive_features(self, attributions):
        """Normalize features by clipping negative values to 0 and normalizing to sum to 1."""
        normalized = {}
        for path in ['baseline', 'unmasked', 'masked']:
            # Clip negative values to 0 and flatten
            attr = np.clip(attributions[path].numpy(), 0, None).flatten()
            # Normalize to sum to 1
            attr = attr / (np.sum(attr) + 1e-10)
            normalized[path] = attr
        return normalized

    def _normalize_all_features(self, attributions):
        """Normalize features by taking absolute values and normalizing to sum to 1."""
        normalized = {}
        for path in ['baseline', 'unmasked', 'masked']:
            # Take absolute values and flatten
            attr = np.abs(attributions[path].numpy()).flatten()
            # Normalize to sum to 1
            attr = attr / (np.sum(attr) + 1e-10)
            normalized[path] = attr
        return normalized

    def _run_feature_analysis(self, attributions, save_dir, is_correct_only=False, use_abs=False):
        """Run feature analysis for all samples or correct samples only with specified normalization method."""
        print(f"\nRunning feature analysis for {'correct' if is_correct_only else 'all'} samples using {'absolute' if use_abs else 'positive'} values...")
        
        # Determine folder name based on parameters
        folder_name = 'all_feature_correct' if (is_correct_only and use_abs) else \
                     'all_feature' if use_abs else \
                     'positive_feature_correct' if is_correct_only else \
                     'positive_feature'
        
        analysis_dir = os.path.join(save_dir, folder_name)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Choose normalization method
        normalize_func = self._normalize_all_features if use_abs else self._normalize_positive_features
        
        # Run analysis
        self.analyze_shap_comparison(attributions, analysis_dir, normalize_func)
        self.analyze_feature_importance(attributions, analysis_dir, normalize_func)

    def run(self, save_dir):
        """Run complete IG analysis pipeline with new folder structure"""
        print("\nRunning SHAP visualization with new folder structure...")
        
        # Calculate attributions
        attributions = self.calculate_attributions()
        
        # Step 1: Run sample visualizations
        self._run_sample_visualization(attributions, save_dir)
        
        # Step 2: Run analysis for all samples with both normalization methods
        self._run_feature_analysis(attributions, save_dir, is_correct_only=False, use_abs=False)  # positive_feature
        self._run_feature_analysis(attributions, save_dir, is_correct_only=False, use_abs=True)   # all_feature
        
        # Step 3: Run analysis for correct samples with both normalization methods
        correct_predictions = self._get_correct_predictions(self.background[0], self.background[1])
        correct_attributions = self._filter_attributions(attributions, correct_predictions['all_correct'])
        
        # Save prediction statistics
        stats = self._save_prediction_stats(correct_predictions, 
                                          os.path.join(save_dir, 'positive_feature_correct'))
        
        # Only proceed if we have correct samples
        if correct_attributions['baseline'].size(0) > 0:
            self._run_feature_analysis(correct_attributions, save_dir, is_correct_only=True, use_abs=False)  # positive_feature_correct
            self._run_feature_analysis(correct_attributions, save_dir, is_correct_only=True, use_abs=True)   # all_feature_correct
        else:
            print("Warning: No samples were correctly predicted by all three paths") 