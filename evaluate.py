import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from metric import (
    load_models,
    evaluate_accuracy,
    evaluate_loss,
    compute_auroc,
    plot_metrics,
    plot_auroc_curves,
    analyze_shap_extremes,
    analyze_feature_variance,
    analyze_feature_space,
    save_feature_variance_results,
    plot_feature_variance,
    per_class_feature_stats,
    compute_feature_overlap,
    plot_per_class_heatmap,
    plot_overlap_matrix,
    plot_mean_per_class_stats,
    run_shap_visualization,
    evaluate_and_plot_models,
    run_feature_analysis_for_split,
    evaluate_robustness
)
# from original_cifar10_cnn import Classifier, MaskGenerator # Removed
from gtsrb_original import GTSRBModel, MaskGenerator # Added
from interpretable_model import InterpretableModel
from shap_visualizer import SHAPVisualizer
import json
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from gtsrb_finetune import GTSRBDataset # Added

def load_gtsrb_data(data_dir: str, batch_size: int = 64):
    """Load GTSRB dataset with appropriate transforms and splits"""
    # GTSRB has 43 classes, names are not typically used directly like CIFAR-10
    
    # Simple transform with only normalization (NO augmentation for evaluation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # GTSRB uses this in finetune
    ])
    
    # GTSRBDataset handles its own data path construction and download is not applicable here
    # It expects root_dir to be the parent of 'GTSRB/GTSRB/...'
    # e.g. if data is in '../../data/GTSRB/GTSRB', then root_dir is '../../data'

    # Load datasets using GTSRBDataset
    train_dataset_full = GTSRBDataset(
        root_dir=data_dir, 
        train=True, 
        transform=transform
    )
    
    test_dataset = GTSRBDataset(
        root_dir=data_dir, 
        train=False, 
        transform=transform
    )
    
    # Split training data into train and validation (consistent with gtsrb_finetune.py and original evaluate_cifar10.py)
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset_full, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=False,  # Shuffle False for consistency in evaluation, though train usually True
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def main(experiment_path: str, pretrained_path: str, interpretable_path: str, 
         radial_radius: float, radial_decay: float, upper_mask_level_threshold: float):
    """
    Evaluate models and generate metrics for GTSRB.
    
    Args:
        experiment_path: Path to the experiment folder
        pretrained_path: Path to the pretrained model (GTSRBModel state dict)
        interpretable_path: Path to the interpretable model checkpoint (InterpretableModel state dict or full model)
        radial_radius: Radius for radial mask (will be converted to integer)
        radial_decay: Decay rate for radial mask
        upper_mask_level_threshold: Upper threshold for mask level
    """
    # Configuration
    data_dir = '../../data'  # Consistent with gtsrb_finetune.py
    base_dir = './logs' # Not directly used here, but for context
    num_classes = 43 # GTSRB has 43 classes
    
    # Construct paths
    metrics_dir = os.path.join(experiment_path, 'eval_results')
    
    # Create metrics directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = load_gtsrb_data(data_dir, batch_size=64) # Default batch size
    
    # Create components for interpretable model
    mask_generator = MaskGenerator() # From gtsrb_original
    classifier = GTSRBModel(num_classes=num_classes) # From gtsrb_original
    
    # Load models using the load_models function
    print(f"\nLoading models from checkpoints:")
    print(f"Pretrained model path: {pretrained_path}")
    print(f"Interpretable model path: {interpretable_path}")
    
    # Convert radial_radius to integer
    radial_radius_int = int(radial_radius)
    
    pretrained_model, interpretable_model = load_models(
        pretrained_path=pretrained_path,
        interpretable_path=interpretable_path,
        device=device,
        pretrained_model_class=GTSRBModel, # Pass GTSRBModel class
        interpretable_model_class=lambda: InterpretableModel( # Lambda to construct interpretable model
            mask_generator=mask_generator,
            classifier=classifier, # Use GTSRBModel instance here
            num_classes=num_classes,
            radial_radius=radial_radius_int,  # Use the integer version
            radial_decay=radial_decay,
            upper_mask_level_threshold=upper_mask_level_threshold
        )
    )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # 1. SHAP Visualization
    print("\nRunning SHAP visualization...")
    shap_dir = os.path.join(metrics_dir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)
    
    # Run SHAP visualization for each dataset
    # SHAPVisualizer might need num_classes if it uses it for class names, but seems to get labels directly
    run_shap_visualization(pretrained_model, interpretable_model, train_loader, device, os.path.join(shap_dir, 'train'))
    run_shap_visualization(pretrained_model, interpretable_model, val_loader, device, os.path.join(shap_dir, 'val'))
    run_shap_visualization(pretrained_model, interpretable_model, test_loader, device, os.path.join(shap_dir, 'test'))
    
    # 2. Model Evaluation and AUROC/Metrics Plots
    print("\nEvaluating models and generating AUROC/metrics plots...")
    metrics, roc_curves = evaluate_and_plot_models(
        pretrained_model, interpretable_model,
        train_loader, val_loader, test_loader,
        device, criterion, metrics_dir,
        num_classes=num_classes # Pass num_classes here
    )
    
    # 3. Feature Analysis
    print("\nRunning feature analysis...")
    feature_analysis_dir = os.path.join(metrics_dir, 'feature_analysis')
    os.makedirs(feature_analysis_dir, exist_ok=True)
    
    # Run feature analysis for each dataset
    run_feature_analysis_for_split(pretrained_model, interpretable_model, train_loader, device, 'train', os.path.join(feature_analysis_dir, 'train'), num_classes=num_classes)
    run_feature_analysis_for_split(pretrained_model, interpretable_model, val_loader, device, 'val', os.path.join(feature_analysis_dir, 'val'), num_classes=num_classes)
    run_feature_analysis_for_split(pretrained_model, interpretable_model, test_loader, device, 'test', os.path.join(feature_analysis_dir, 'test'), num_classes=num_classes)
    
    # 4. Robustness Evaluation
    print("\nEvaluating model robustness...")
    evaluate_robustness(pretrained_model, interpretable_model, train_loader, val_loader, test_loader, device, metrics_dir, num_classes=num_classes)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GTSRB models') # Updated description
    parser.add_argument('--experiment_path', type=str, required=True,
                      help='Path to the experiment folder')
    parser.add_argument('--pretrained_path', type=str, required=True,
                      help='Path to the pretrained GTSRB model') # Updated help
    parser.add_argument('--interpretable_path', type=str, required=True,
                      help='Path to the interpretable model checkpoint')
    parser.add_argument('--radial_radius', type=float, required=True,
                      help='Radius for radial mask')
    parser.add_argument('--radial_decay', type=float, required=True,
                      help='Decay rate for radial mask')
    parser.add_argument('--upper_mask_level_threshold', type=float, required=True,
                      help='Upper threshold for mask level')
    
    args = parser.parse_args()
    
    main(
        experiment_path=args.experiment_path,
        pretrained_path=args.pretrained_path,
        interpretable_path=args.interpretable_path,
        radial_radius=args.radial_radius,
        radial_decay=args.radial_decay,
        upper_mask_level_threshold=args.upper_mask_level_threshold
    ) 