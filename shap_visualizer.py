import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy.stats as stats
from captum.attr import GradientShap
from matplotlib.colors import LinearSegmentedColormap
from toolkits import apply_radial_mask

class SHAPVisualizer:
    def __init__(self, pretrained_model, interpretable_model, data_loader, device, num_samples=50, num_visualize=50):
        """
        Initialize SHAP visualizer.
        
        Args:
            pretrained_model: The pretrained model for baseline SHAP values
            interpretable_model: The interpretable model for masked/unmasked paths
            data_loader: DataLoader for the dataset
            device: Device to run computations on
            num_samples: Number of samples to use for SHAP calculation (increased to 50)
            num_visualize: Number of samples to visualize (increased to 50)
        """
        self.pretrained_model = pretrained_model
        self.interpretable_model = interpretable_model
        self.data_loader = data_loader
        self.device = device
        self.num_samples = num_samples
        self.num_visualize = num_visualize
        
        # Create background dataset for SHAP
        self.background = self._create_background()
        
    def _create_background(self):
        """Create background dataset for SHAP calculation"""
        print("Creating background dataset...")
        # Sample images from the dataset
        all_images = []
        all_labels = []
        
        # Get one batch
        for images, labels in self.data_loader:
            all_images.append(images)
            all_labels.append(labels)
            break  # Only take first batch
        
        # Take only the number of samples we need
        background_images = torch.cat(all_images, dim=0)[:self.num_samples]
        background_labels = torch.cat(all_labels, dim=0)[:self.num_samples]
        
        print(f"Selected {len(background_images)} samples for SHAP analysis")
        return background_images, background_labels
    
    def _model_predict(self, x):
        """Wrapper for pretrained model prediction for SHAP - baseline path"""
        # Use pretrained model directly
        logits = self.pretrained_model(x)
        return logits
    
    def _model_predict_masked(self, x):
        """Wrapper for masked model prediction for SHAP - direct attribution to masked input"""
        # Get the masked input directly from the model's forward pass
        outputs = self.interpretable_model(x)
        masked_x = outputs['masked_x']
        # Get predictions using the classifier on masked input
        logits = self.interpretable_model.classifier(masked_x)
        return logits
    
    def _model_predict_unmasked(self, x):
        """Wrapper for unmasked model prediction for SHAP"""
        # For unmasked path, just use the classifier directly
        logits = self.interpretable_model.classifier(x)
        return logits
    
    def calculate_shap_values(self):
        """Calculate SHAP values for all paths"""
        print("\nCalculating SHAP values...")
        
        # Initialize SHAP explainers
        background_images = self.background[0].to(self.device)
        background_labels = self.background[1].to(self.device)
        
        # Create baseline (black image)
        baseline = torch.zeros_like(background_images).to(self.device)
        
        # Baseline path - attribution to original input using pretrained model
        print("\nCalculating baseline SHAP values (pretrained model)...")
        baseline_explainer = GradientShap(self._model_predict)
        shap_values_baseline = baseline_explainer.attribute(
            background_images, 
            baseline,
            target=background_labels,
            n_samples=10  # Number of samples for gradient estimation
        )
        
        # Unmasked path - attribution to original input
        print("\nCalculating unmasked SHAP values...")
        unmasked_explainer = GradientShap(self._model_predict_unmasked)
        shap_values_unmasked = unmasked_explainer.attribute(
            background_images,
            baseline,
            target=background_labels,
            n_samples=10
        )
        
        # Masked path - attribution to masked input
        print("\nCalculating masked SHAP values...")
        masked_explainer = GradientShap(self._model_predict_masked)
        shap_values_masked = masked_explainer.attribute(
            background_images,
            baseline,
            target=background_labels,
            n_samples=10
        )
        
        return {
            'baseline': shap_values_baseline,
            'unmasked': shap_values_unmasked,
            'masked': shap_values_masked
        }
    
    def analyze_shap_comparison(self, shap_values, save_dir):
        """Analyze and compare SHAP values between baseline, unmasked, and masked paths"""
        print("\nAnalyzing SHAP value distributions...")
        
        # Create analysis directory
        analysis_dir = os.path.join(save_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Calculate statistics for each sample
        baseline_entropy = []
        unmasked_entropy = []
        masked_entropy = []
        baseline_std = []
        unmasked_std = []
        masked_std = []
        
        for i in tqdm(range(self.num_samples), desc="Analyzing samples"):
            # Get SHAP values for the true class
            true_label = self.background[1][i]
            baseline_shap = np.abs(shap_values['baseline'][i].cpu().numpy())
            unmasked_shap = np.abs(shap_values['unmasked'][i].cpu().numpy())
            masked_shap = np.abs(shap_values['masked'][i].cpu().numpy())
            
            # Calculate entropy of SHAP values
            baseline_entropy.append(stats.entropy(baseline_shap.flatten()))
            unmasked_entropy.append(stats.entropy(unmasked_shap.flatten()))
            masked_entropy.append(stats.entropy(masked_shap.flatten()))
            
            # Calculate standard deviation
            baseline_std.append(np.std(baseline_shap))
            unmasked_std.append(np.std(unmasked_shap))
            masked_std.append(np.std(masked_shap))
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        
        # Entropy comparison
        plt.subplot(1, 3, 1)
        plt.boxplot([baseline_entropy, unmasked_entropy, masked_entropy], 
                   labels=['Baseline', 'Unmasked', 'Masked'])
        plt.title('SHAP Value Entropy Comparison')
        plt.ylabel('Entropy')
        
        # Standard deviation comparison
        plt.subplot(1, 3, 2)
        plt.boxplot([baseline_std, unmasked_std, masked_std], 
                   labels=['Baseline', 'Unmasked', 'Masked'])
        plt.title('SHAP Value Standard Deviation Comparison')
        plt.ylabel('Standard Deviation')
        
        # Plot entropy differences
        plt.subplot(1, 3, 3)
        unmasked_diff = np.array(unmasked_entropy) - np.array(baseline_entropy)
        masked_diff = np.array(masked_entropy) - np.array(baseline_entropy)
        plt.boxplot([unmasked_diff, masked_diff],
                   labels=['Unmasked - Baseline', 'Masked - Baseline'])
        plt.title('Entropy Difference from Baseline')
        plt.ylabel('Entropy Difference')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'shap_comparison.png'))
        plt.close()
        
        # Calculate and save statistics
        stats_file = os.path.join(analysis_dir, 'shap_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("SHAP Value Analysis\n")
            f.write("==================\n\n")
            
            f.write("Entropy Statistics:\n")
            f.write(f"Baseline Mean: {np.mean(baseline_entropy):.4f}\n")
            f.write(f"Baseline Std: {np.std(baseline_entropy):.4f}\n")
            f.write(f"Unmasked Mean: {np.mean(unmasked_entropy):.4f}\n")
            f.write(f"Unmasked Std: {np.std(unmasked_entropy):.4f}\n")
            f.write(f"Masked Mean: {np.mean(masked_entropy):.4f}\n")
            f.write(f"Masked Std: {np.std(masked_entropy):.4f}\n")
            f.write(f"Entropy Difference (Unmasked - Baseline): {np.mean(unmasked_entropy) - np.mean(baseline_entropy):.4f}\n")
            f.write(f"Entropy Difference (Masked - Baseline): {np.mean(masked_entropy) - np.mean(baseline_entropy):.4f}\n\n")
            
            f.write("Standard Deviation Statistics:\n")
            f.write(f"Baseline Mean: {np.mean(baseline_std):.4f}\n")
            f.write(f"Baseline Std: {np.std(baseline_std):.4f}\n")
            f.write(f"Unmasked Mean: {np.mean(unmasked_std):.4f}\n")
            f.write(f"Unmasked Std: {np.std(unmasked_std):.4f}\n")
            f.write(f"Masked Mean: {np.mean(masked_std):.4f}\n")
            f.write(f"Masked Std: {np.std(masked_std):.4f}\n")
            f.write(f"Std Difference (Unmasked - Baseline): {np.mean(unmasked_std) - np.mean(baseline_std):.4f}\n")
            f.write(f"Std Difference (Masked - Baseline): {np.mean(masked_std) - np.mean(baseline_std):.4f}\n")
        
        print(f"Analysis saved to {analysis_dir}")
    
    def visualize(self, shap_values, save_dir):
        """Visualize SHAP values and masks for selected samples"""
        print("\nVisualizing results...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Select samples to visualize
        background_images = self.background[0].to(self.device)
        background_labels = self.background[1]
        
        # Get model outputs for selected samples
        with torch.no_grad():
            outputs = self.interpretable_model(background_images)
            soft_masks = outputs['soft_mask']
            radial_masks = outputs['radial_mask']
            binary_masks = outputs['binary_mask']
            masked_images = outputs['masked_input']
        
        # Create custom colormap
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
        
        # Visualize each sample
        for i in tqdm(range(min(self.num_visualize, self.num_samples)), desc="Generating visualizations"):
            fig, axs = plt.subplots(2, 5, figsize=(20, 8))
            
            # Original image
            img = background_images[i].cpu().numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # Unnormalize
            
            # Row 1: Original image and SHAP values
            axs[0, 0].imshow(img)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')
            
            # SHAP values for baseline
            shap_img = np.abs(shap_values['baseline'][i].cpu().numpy()).sum(0)
            shap_img = shap_img / (np.max(np.abs(shap_img)) + 1e-10)
            axs[0, 1].imshow(shap_img, cmap=cmap)
            axs[0, 1].set_title('Baseline SHAP')
            axs[0, 1].axis('off')
            
            # SHAP values for unmasked
            shap_img = np.abs(shap_values['unmasked'][i].cpu().numpy()).sum(0)
            shap_img = shap_img / (np.max(np.abs(shap_img)) + 1e-10)
            axs[0, 2].imshow(shap_img, cmap=cmap)
            axs[0, 2].set_title('Unmasked SHAP')
            axs[0, 2].axis('off')
            
            # SHAP values for masked
            shap_img = np.abs(shap_values['masked'][i].cpu().numpy()).sum(0)
            shap_img = shap_img / (np.max(np.abs(shap_img)) + 1e-10)
            axs[0, 3].imshow(shap_img, cmap=cmap)
            axs[0, 3].set_title('Masked SHAP')
            axs[0, 3].axis('off')
            
            # Add predictions
            with torch.no_grad():
                baseline_pred = torch.argmax(self.pretrained_model(background_images[i:i+1])).item()
                unmasked_pred = torch.argmax(outputs['unmasked_logits'][i:i+1]).item()
                masked_pred = torch.argmax(outputs['masked_logits'][i:i+1]).item()
            
            axs[0, 4].text(0.5, 0.5, 
                          f'Baseline: {baseline_pred}\n'
                          f'Unmasked: {unmasked_pred}\n'
                          f'Masked: {masked_pred}\n'
                          f'True: {background_labels[i]}',
                          ha='center', va='center')
            axs[0, 4].axis('off')
            
            # Row 2: Masked image and masks
            masked_img = masked_images[i].cpu().numpy().transpose((1, 2, 0))
            masked_img = masked_img * 0.5 + 0.5  # Unnormalize
            
            axs[1, 0].imshow(masked_img)
            axs[1, 0].set_title('Masked Image')
            axs[1, 0].axis('off')
            
            # Soft mask
            soft_mask = soft_masks[i].cpu().squeeze()
            axs[1, 1].imshow(soft_mask, cmap='viridis')
            axs[1, 1].set_title('Soft Mask')
            axs[1, 1].axis('off')
            
            # Radial mask
            radial_mask = radial_masks[i].cpu().squeeze()
            axs[1, 2].imshow(radial_mask, cmap='viridis')
            axs[1, 2].set_title('Radial Mask')
            axs[1, 2].axis('off')
            
            # Binary mask
            binary_mask = binary_masks[i].cpu().squeeze()
            axs[1, 3].imshow(binary_mask, cmap='binary')
            axs[1, 3].set_title('Binary Mask')
            axs[1, 3].axis('off')
            
            # Add mask statistics
            axs[1, 4].text(0.5, 0.5, 
                          f'Soft Mask Mean: {soft_mask.mean():.3f}\n'
                          f'Radial Mask Mean: {radial_mask.mean():.3f}\n'
                          f'Binary Mask Mean: {binary_mask.mean():.3f}',
                          ha='center', va='center')
            axs[1, 4].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {save_dir}")
    
    def analyze_feature_importance(self, shap_values, save_dir, importance_threshold=0.1, unimportance_threshold=0.01):
        """Analyze and visualize important vs unimportant features and input pixels based on SHAP values"""
        print("\nAnalyzing feature and input importance...")
        
        # Create analysis directory
        analysis_dir = os.path.join(save_dir, 'feature_importance')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Initialize counters for both feature and input analysis
        important_features = []
        unimportant_features = []
        important_inputs = []
        unimportant_inputs = []
        
        # For distribution analysis
        all_shap_values = {
            'baseline': {'features': [], 'inputs': []},
            'unmasked': {'features': [], 'inputs': []},
            'masked': {'features': [], 'inputs': []}
        }
        
        # For normalized importance analysis
        normalized_important_features = []
        normalized_unimportant_features = []
        normalized_important_inputs = []
        normalized_unimportant_inputs = []
        
        # Normalized thresholds (since values will be divided by total sum)
        norm_importance_threshold = 0.001  # 0.1% of total attribution
        norm_unimportance_threshold = 0.0001  # 0.01% of total attribution
        
        for i in tqdm(range(self.num_samples), desc="Analyzing samples"):
            # Get SHAP values for the true class
            true_label = self.background[1][i]
            baseline_shap = np.abs(shap_values['baseline'][i].cpu().numpy())
            unmasked_shap = np.abs(shap_values['unmasked'][i].cpu().numpy())
            masked_shap = np.abs(shap_values['masked'][i].cpu().numpy())
            
            # Get total sums for normalization
            baseline_sum = np.sum(baseline_shap)
            unmasked_sum = np.sum(unmasked_shap)
            masked_sum = np.sum(masked_shap)
            
            # Normalize SHAP values
            baseline_norm = baseline_shap / (baseline_sum + 1e-10)
            unmasked_norm = unmasked_shap / (unmasked_sum + 1e-10)
            masked_norm = masked_shap / (masked_sum + 1e-10)
            
            # Feature importance (MLP features) - Raw
            important_baseline_feat = np.sum(baseline_shap > importance_threshold)
            unimportant_baseline_feat = np.sum(baseline_shap < unimportance_threshold)
            important_unmasked_feat = np.sum(unmasked_shap > importance_threshold)
            unimportant_unmasked_feat = np.sum(unmasked_shap < unimportance_threshold)
            important_masked_feat = np.sum(masked_shap > importance_threshold)
            unimportant_masked_feat = np.sum(masked_shap < unimportance_threshold)
            
            # Feature importance - Normalized
            norm_important_baseline_feat = np.sum(baseline_norm > norm_importance_threshold)
            norm_unimportant_baseline_feat = np.sum(baseline_norm < norm_unimportance_threshold)
            norm_important_unmasked_feat = np.sum(unmasked_norm > norm_importance_threshold)
            norm_unimportant_unmasked_feat = np.sum(unmasked_norm < norm_unimportance_threshold)
            norm_important_masked_feat = np.sum(masked_norm > norm_importance_threshold)
            norm_unimportant_masked_feat = np.sum(masked_norm < norm_unimportance_threshold)
            
            # Input pixel importance - Raw
            important_baseline_inp = np.sum(baseline_shap > importance_threshold)
            unimportant_baseline_inp = np.sum(baseline_shap < unimportance_threshold)
            important_unmasked_inp = np.sum(unmasked_shap > importance_threshold)
            unimportant_unmasked_inp = np.sum(unmasked_shap < unimportance_threshold)
            important_masked_inp = np.sum(masked_shap > importance_threshold)
            unimportant_masked_inp = np.sum(masked_shap < unimportance_threshold)
            
            # Input pixel importance - Normalized
            norm_important_baseline_inp = np.sum(baseline_norm > norm_importance_threshold)
            norm_unimportant_baseline_inp = np.sum(baseline_norm < norm_unimportance_threshold)
            norm_important_unmasked_inp = np.sum(unmasked_norm > norm_importance_threshold)
            norm_unimportant_unmasked_inp = np.sum(unmasked_norm < norm_unimportance_threshold)
            norm_important_masked_inp = np.sum(masked_norm > norm_importance_threshold)
            norm_unimportant_masked_inp = np.sum(masked_norm < norm_unimportance_threshold)
            
            # Store raw feature counts
            important_features.append({
                'baseline': important_baseline_feat,
                'unmasked': important_unmasked_feat,
                'masked': important_masked_feat
            })
            unimportant_features.append({
                'baseline': unimportant_baseline_feat,
                'unmasked': unimportant_unmasked_feat,
                'masked': unimportant_masked_feat
            })
            
            # Store normalized feature counts
            normalized_important_features.append({
                'baseline': norm_important_baseline_feat,
                'unmasked': norm_important_unmasked_feat,
                'masked': norm_important_masked_feat
            })
            normalized_unimportant_features.append({
                'baseline': norm_unimportant_baseline_feat,
                'unmasked': norm_unimportant_unmasked_feat,
                'masked': norm_unimportant_masked_feat
            })
            
            # Store raw input counts
            important_inputs.append({
                'baseline': important_baseline_inp,
                'unmasked': important_unmasked_inp,
                'masked': important_masked_inp
            })
            unimportant_inputs.append({
                'baseline': unimportant_baseline_inp,
                'unmasked': unimportant_unmasked_inp,
                'masked': unimportant_masked_inp
            })
            
            # Store normalized input counts
            normalized_important_inputs.append({
                'baseline': norm_important_baseline_inp,
                'unmasked': norm_important_unmasked_inp,
                'masked': norm_important_masked_inp
            })
            normalized_unimportant_inputs.append({
                'baseline': norm_unimportant_baseline_inp,
                'unmasked': norm_unimportant_unmasked_inp,
                'masked': norm_unimportant_masked_inp
            })
            
            # Store all SHAP values for distribution analysis
            all_shap_values['baseline']['features'].extend(baseline_shap.flatten())
            all_shap_values['unmasked']['features'].extend(unmasked_shap.flatten())
            all_shap_values['masked']['features'].extend(masked_shap.flatten())
            all_shap_values['baseline']['inputs'].extend(baseline_shap.flatten())
            all_shap_values['unmasked']['inputs'].extend(unmasked_shap.flatten())
            all_shap_values['masked']['inputs'].extend(masked_shap.flatten())
        
        def plot_importance_comparison(important_data, unimportant_data, title_prefix, threshold_high, threshold_low, save_path):
            """Helper function to plot importance comparisons"""
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Important features plot
            baseline_important = [f['baseline'] for f in important_data]
            unmasked_important = [f['unmasked'] for f in important_data]
            masked_important = [f['masked'] for f in important_data]
            ax1.boxplot([baseline_important, unmasked_important, masked_important],
                      labels=['Baseline', 'Unmasked', 'Masked'])
            ax1.set_title(f'Important {title_prefix} (SHAP > {threshold_high})')
            ax1.set_ylabel('Number of Important Features')
            
            # Unimportant features plot
            baseline_unimportant = [f['baseline'] for f in unimportant_data]
            unmasked_unimportant = [f['unmasked'] for f in unimportant_data]
            masked_unimportant = [f['masked'] for f in unimportant_data]
            ax2.boxplot([baseline_unimportant, unmasked_unimportant, masked_unimportant],
                      labels=['Baseline', 'Unmasked', 'Masked'])
            ax2.set_title(f'Unimportant {title_prefix} (SHAP < {threshold_low})')
            ax2.set_ylabel('Number of Unimportant Features')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        # Plot raw MLP Feature importance comparison
        plot_importance_comparison(
            important_features, 
            unimportant_features,
            'MLP Features',
            importance_threshold,
            unimportance_threshold,
            os.path.join(analysis_dir, 'mlp_feature_importance.png')
        )
        
        # Plot normalized MLP Feature importance comparison
        plot_importance_comparison(
            normalized_important_features,
            normalized_unimportant_features,
            'MLP Features (Normalized)',
            norm_importance_threshold,
            norm_unimportance_threshold,
            os.path.join(analysis_dir, 'mlp_feature_importance_normalized.png')
        )
        
        # Plot raw Input Pixel importance comparison
        plot_importance_comparison(
            important_inputs,
            unimportant_inputs,
            'Input Pixels',
            importance_threshold,
            unimportance_threshold,
            os.path.join(analysis_dir, 'input_pixel_importance.png')
        )
        
        # Plot normalized Input Pixel importance comparison
        plot_importance_comparison(
            normalized_important_inputs,
            normalized_unimportant_inputs,
            'Input Pixels (Normalized)',
            norm_importance_threshold,
            norm_unimportance_threshold,
            os.path.join(analysis_dir, 'input_pixel_importance_normalized.png')
        )
        
        # Plot 3: SHAP value distributions - separate MLP and Input plots with aligned axes
        # Find max SHAP value for proper x-axis scaling
        max_mlp_shap = max(
            np.max(all_shap_values['baseline']['features']),
            np.max(all_shap_values['unmasked']['features']),
            np.max(all_shap_values['masked']['features'])
        )
        max_input_shap = max(
            np.max(all_shap_values['baseline']['inputs']),
            np.max(all_shap_values['unmasked']['inputs']),
            np.max(all_shap_values['masked']['inputs'])
        )
        
        # Print information about number of samples
        num_feature_values = len(all_shap_values['baseline']['features'])
        num_input_values = len(all_shap_values['baseline']['inputs'])
        print(f"\nSHAP Distribution Statistics:")
        print(f"Number of samples analyzed: {self.num_samples}")
        print(f"Total MLP feature values per path: {num_feature_values:,} (from {self.num_samples} samples)")
        print(f"Total input pixel values per path: {num_input_values:,} (from {self.num_samples} samples)")
        
        # Add sample size to plot titles
        mlp_title = f'MLP Feature SHAP Value Distribution\n(from {self.num_samples} samples, {num_feature_values:,} total values)'
        input_title = f'Input Pixel SHAP Value Distribution\n(from {self.num_samples} samples, {num_input_values:,} total values)'
        
        # Round up to nearest integer for x-axis limit
        max_mlp_shap = np.ceil(max_mlp_shap)
        max_input_shap = np.ceil(max_input_shap)
        
        # Plot MLP Feature distributions
        fig_mlp, axs_mlp = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
        fig_mlp.suptitle(mlp_title, fontsize=14)
        
        paths = ['baseline', 'unmasked', 'masked']
        colors = ['skyblue', 'orange', 'lightgreen']
        
        # First pass to find max density for y-axis scaling
        max_density = 0
        for path in paths:
            values = all_shap_values[path]['features']
            hist, _ = np.histogram(values, bins=50, density=True)
            max_density = max(max_density, np.max(hist))
        
        # Plot with consistent y-axis
        for idx, (path, color) in enumerate(zip(paths, colors)):
            values = all_shap_values[path]['features']
            axs_mlp[idx].hist(values, bins=50, color=color, alpha=0.7, density=True)
            axs_mlp[idx].set_ylabel(f'{path.capitalize()}\nDensity')
            # Add mean and count
            mean_val = np.mean(values)
            axs_mlp[idx].axvline(x=mean_val, color='red', linestyle='--', alpha=0.5,
                                label=f'Mean: {mean_val:.3f} (n={len(values):,})')
            axs_mlp[idx].legend()
            # Set y-axis limit
            axs_mlp[idx].set_ylim(0, max_density * 1.1)  # Add 10% padding
        
        axs_mlp[-1].set_xlabel('SHAP Value')
        plt.xlim(0, max_mlp_shap)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'mlp_shap_distributions.png'))
        plt.close()
        
        # Plot Input Pixel distributions
        fig_input, axs_input = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig_input.suptitle(input_title, fontsize=14)
        
        for idx, (path, color) in enumerate(zip(paths, colors)):
            values = all_shap_values[path]['inputs']
            axs_input[idx].hist(values, bins=50, color=color, alpha=0.7, density=True)
            axs_input[idx].set_ylabel(f'{path.capitalize()}\nDensity')
            # Add mean and count
            mean_val = np.mean(values)
            axs_input[idx].axvline(x=mean_val, color='red', linestyle='--', alpha=0.5,
                                  label=f'Mean: {mean_val:.3f} (n={len(values):,})')
            axs_input[idx].legend()
        
        axs_input[-1].set_xlabel('SHAP Value')
        plt.xlim(0, max_input_shap)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'input_shap_distributions.png'))
        plt.close()
        
        # Add normalized distribution plots
        def plot_normalized_distributions(values_dict, title, save_path, paths, colors):
            fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
            fig.suptitle(f'Normalized {title}\n(Sum of attributions = 1 for each path)', fontsize=14)
            
            # First pass to find max density for y-axis scaling
            max_density = 0
            normalized_values = {}
            
            for path in paths:
                # Get values and normalize by total sum
                values = np.array(values_dict[path])
                total_sum = np.sum(np.abs(values))  # Use absolute values for normalization
                normalized_values[path] = values / (total_sum + 1e-10)  # Add epsilon to avoid division by zero
                
                # Calculate max density for consistent y-axis
                hist, _ = np.histogram(normalized_values[path], bins=50, density=True)
                max_density = max(max_density, np.max(hist))
            
            # Plot with consistent y-axis
            for idx, (path, color) in enumerate(zip(paths, colors)):
                values = normalized_values[path]
                axs[idx].hist(values, bins=50, color=color, alpha=0.7, density=True)
                axs[idx].set_ylabel(f'{path.capitalize()}\nDensity')
                
                # Add statistics
                mean_val = np.mean(values)
                median_val = np.median(values)
                axs[idx].axvline(x=mean_val, color='red', linestyle='--', alpha=0.5,
                                label=f'Mean: {mean_val:.2e}\nMedian: {median_val:.2e}\n(n={len(values):,})')
                axs[idx].legend()
                axs[idx].set_ylim(0, max_density * 1.1)  # Add 10% padding
            
            axs[-1].set_xlabel('Normalized SHAP Value (Attribution / Total Attribution)')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        # Plot normalized MLP Feature distributions
        plot_normalized_distributions(
            {path: all_shap_values[path]['features'] for path in paths},
            'MLP Feature SHAP Value Distribution',
            os.path.join(analysis_dir, 'mlp_shap_distributions_normalized.png'),
            paths,
            colors
        )
        
        # Plot normalized Input Pixel distributions
        plot_normalized_distributions(
            {path: all_shap_values[path]['inputs'] for path in paths},
            'Input Pixel SHAP Value Distribution',
            os.path.join(analysis_dir, 'input_shap_distributions_normalized.png'),
            paths,
            colors
        )
        
        # Add sample size information to statistics file
        stats_file = os.path.join(analysis_dir, 'feature_importance_statistics.txt')
        with open(stats_file, 'a') as f:
            f.write("\nSample Size Information:\n")
            f.write("=====================\n")
            f.write(f"Number of samples analyzed: {self.num_samples}\n")
            f.write(f"Total MLP feature values per path: {num_feature_values:,}\n")
            f.write(f"Total input pixel values per path: {num_input_values:,}\n")
            
            # Add normalization statistics
            f.write("\nNormalized Distribution Statistics:\n")
            f.write("==============================\n")
            for path in paths:
                f.write(f"\n{path.capitalize()} Path:\n")
                for feature_type in ['features', 'inputs']:
                    values = np.array(all_shap_values[path][feature_type])
                    total_sum = np.sum(np.abs(values))
                    normalized_values = values / (total_sum + 1e-10)
                    f.write(f"{feature_type.capitalize()}:\n")
                    f.write(f"  Mean normalized value: {np.mean(normalized_values):.2e}\n")
                    f.write(f"  Median normalized value: {np.median(normalized_values):.2e}\n")
                    f.write(f"  Std normalized value: {np.std(normalized_values):.2e}\n")

    def run(self, save_dir):
        """Run complete SHAP analysis pipeline"""
        shap_values = self.calculate_shap_values()
        self.analyze_shap_comparison(shap_values, save_dir)
        self.analyze_feature_importance(shap_values, save_dir)
        self.visualize(shap_values, save_dir) 