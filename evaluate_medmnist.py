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
    analyze_feature_variance
)
from original_cifar10_cnn import Classifier, MaskGenerator
from interpretable_model import InterpretableModel
from shap_visualizer import SHAPVisualizer
import json

def load_cifar10_data(data_dir: str, batch_size: int = 64):
    """Load CIFAR-10 dataset with appropriate transforms and splits"""
    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Simple transform with only normalization (NO augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Check if dataset exists locally
    data_path = os.path.join(data_dir, 'cifar-10-batches-py')
    if os.path.exists(data_path):
        print("Found local CIFAR-10 dataset")
        download = False
    else:
        print("Local dataset not found, will download")
        download = True
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=download, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=download, 
        transform=transform
    )
    
    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,  # Enable shuffling for training data
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

def main(experiment_folder: str):
    """
    Evaluate models and generate metrics.
    
    Args:
        experiment_folder: Name of the experiment folder (e.g., 'cifar10_interpretable')
    """
    # Configuration
    data_dir = './data'  # This is the correct path to the data directory
    base_dir = './logs'
    
    # Construct paths
    experiment_dir = os.path.join(base_dir, experiment_folder)
    pretrained_path = './pretrain/cifar10/cifar10cnn/best_model.pth'
    interpretable_path = os.path.join(experiment_dir, 'checkpoints', 'final_epoch.pth')
    metrics_dir = os.path.join(experiment_dir, 'metrics')
    
    # Create metrics directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = load_cifar10_data(data_dir)
    
    # Create components for interpretable model
    mask_generator = MaskGenerator()
    classifier = Classifier(num_classes=10)
    
    # Load models using the load_models function
    print(f"\nLoading models from checkpoints:")
    print(f"Pretrained model path: {pretrained_path}")
    print(f"Interpretable model path: {interpretable_path}")
    
    pretrained_model, interpretable_model = load_models(
        pretrained_path=pretrained_path,
        interpretable_path=interpretable_path,
        device=device,
        pretrained_model_class=Classifier,
        interpretable_model_class=lambda: InterpretableModel(
            mask_generator=mask_generator,
            classifier=classifier,
            num_classes=10,
            radial_radius=1,  # Same as training
            radial_decay=0.2,  # Same as training
            upper_mask_level_threshold=0.8  # Same as training
        )
    )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Run SHAP visualization first
    print("\nRunning SHAP visualization...")
    shap_dir = os.path.join(metrics_dir, 'shap_visualization')
    os.makedirs(shap_dir, exist_ok=True)
    
    # Create SHAP visualizer for each dataset
    for dataset_name, data_loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"\nAnalyzing {dataset_name} set...")
        dataset_dir = os.path.join(shap_dir, dataset_name)
        visualizer = SHAPVisualizer(pretrained_model, interpretable_model, data_loader, device)
        visualizer.run(dataset_dir)
    
    print("\nSHAP visualization complete!")
    
    # Evaluate models
    metrics = {}
    
    # Pretrained model evaluation
    print("\nEvaluating pretrained model...")
    metrics['pretrained_train_acc'] = evaluate_accuracy(pretrained_model, train_loader, device)
    metrics['pretrained_val_acc'] = evaluate_accuracy(pretrained_model, val_loader, device)
    metrics['pretrained_test_acc'] = evaluate_accuracy(pretrained_model, test_loader, device)
    metrics['pretrained_train_loss'] = evaluate_loss(pretrained_model, train_loader, criterion, device)
    metrics['pretrained_val_loss'] = evaluate_loss(pretrained_model, val_loader, criterion, device)
    metrics['pretrained_test_loss'] = evaluate_loss(pretrained_model, test_loader, criterion, device)
    
    # Interpretable model evaluation
    print("\nEvaluating interpretable model...")
    # Unmasked path
    metrics['interpretable_unmasked_train_acc'] = evaluate_accuracy(interpretable_model, train_loader, device, is_interpretable=True)
    metrics['interpretable_unmasked_val_acc'] = evaluate_accuracy(interpretable_model, val_loader, device, is_interpretable=True)
    metrics['interpretable_unmasked_test_acc'] = evaluate_accuracy(interpretable_model, test_loader, device, is_interpretable=True)
    metrics['interpretable_unmasked_train_loss'] = evaluate_loss(interpretable_model, train_loader, criterion, device, is_interpretable=True)
    metrics['interpretable_unmasked_val_loss'] = evaluate_loss(interpretable_model, val_loader, criterion, device, is_interpretable=True)
    metrics['interpretable_unmasked_test_loss'] = evaluate_loss(interpretable_model, test_loader, criterion, device, is_interpretable=True)
    
    # Masked path
    metrics['interpretable_masked_train_acc'] = evaluate_accuracy(interpretable_model, train_loader, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_val_acc'] = evaluate_accuracy(interpretable_model, val_loader, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_test_acc'] = evaluate_accuracy(interpretable_model, test_loader, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_train_loss'] = evaluate_loss(interpretable_model, train_loader, criterion, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_val_loss'] = evaluate_loss(interpretable_model, val_loader, criterion, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_test_loss'] = evaluate_loss(interpretable_model, test_loader, criterion, device, is_interpretable=True, masked_path=True)
    
    # Compute AUROC curves
    roc_curves = {}
    
    # Pretrained model
    roc_curves['Pretrained (Train)'] = compute_auroc(pretrained_model, train_loader, device, num_classes=10)
    roc_curves['Pretrained (Val)'] = compute_auroc(pretrained_model, val_loader, device, num_classes=10)
    roc_curves['Pretrained (Test)'] = compute_auroc(pretrained_model, test_loader, device, num_classes=10)
    
    # Interpretable model - unmasked path
    roc_curves['Interpretable Unmasked (Train)'] = compute_auroc(interpretable_model, train_loader, device, num_classes=10, is_interpretable=True)
    roc_curves['Interpretable Unmasked (Val)'] = compute_auroc(interpretable_model, val_loader, device, num_classes=10, is_interpretable=True)
    roc_curves['Interpretable Unmasked (Test)'] = compute_auroc(interpretable_model, test_loader, device, num_classes=10, is_interpretable=True)
    
    # Interpretable model - masked path
    roc_curves['Interpretable Masked (Train)'] = compute_auroc(interpretable_model, train_loader, device, num_classes=10, is_interpretable=True, masked_path=True)
    roc_curves['Interpretable Masked (Val)'] = compute_auroc(interpretable_model, val_loader, device, num_classes=10, is_interpretable=True, masked_path=True)
    roc_curves['Interpretable Masked (Test)'] = compute_auroc(interpretable_model, test_loader, device, num_classes=10, is_interpretable=True, masked_path=True)
    
    # Plot and save metrics
    plot_metrics(metrics, os.path.join(metrics_dir, 'metrics_comparison.png'))
    plot_auroc_curves(roc_curves, os.path.join(metrics_dir, 'auroc_curves.png'))
    
    # Run feature variance analysis
    print("\nRunning feature variance analysis...")
    
    # Create directory for interpretability analysis
    interpretability_dir = os.path.join(metrics_dir, 'interpretability_analysis')
    os.makedirs(interpretability_dir, exist_ok=True)
    
    # Analyze feature variance for each dataset
    for dataset_name, data_loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"\nAnalyzing feature variance for {dataset_name} set...")
        dataset_dir = os.path.join(interpretability_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Analyze pretrained model
        pretrained_variance = analyze_feature_variance(pretrained_model, data_loader, device)
        
        # Analyze interpretable model
        interpretable_variance = analyze_feature_variance(interpretable_model, data_loader, device, is_interpretable=True)
        
        # Save results
        variance_results = {
            'pretrained': {
                'class_variances': pretrained_variance['class_variances'].tolist(),
                'class_means': pretrained_variance['class_means'].tolist(),
                'overall_variance': pretrained_variance['overall_variance'].item()
            },
            'interpretable': {
                'class_variances': interpretable_variance['class_variances'].tolist(),
                'class_means': interpretable_variance['class_means'].tolist(),
                'overall_variance': interpretable_variance['overall_variance'].item(),
                'masked_class_variances': interpretable_variance['masked_class_variances'].tolist(),
                'masked_class_means': interpretable_variance['masked_class_means'].tolist(),
                'masked_overall_variance': interpretable_variance['masked_overall_variance'].item()
            }
        }
        
        # Save to JSON file
        with open(os.path.join(dataset_dir, 'feature_variance.txt'), 'w') as f:
            f.write(f"Feature Variance Analysis for {dataset_name.upper()} set\n")
            f.write("=" * 50 + "\n\n")
            f.write("Pretrained Model:\n")
            f.write(f"Overall Variance: {pretrained_variance['overall_variance'].item():.4f}\n")
            f.write("Class-wise Variances:\n")
            for i, var in enumerate(pretrained_variance['class_variances']):
                f.write(f"Class {i}: {var.item():.4f}\n")
            f.write("\nInterpretable Model:\n")
            f.write(f"Overall Variance (Unmasked): {interpretable_variance['overall_variance'].item():.4f}\n")
            f.write(f"Overall Variance (Masked): {interpretable_variance['masked_overall_variance'].item():.4f}\n")
            f.write("Class-wise Variances (Unmasked):\n")
            for i, var in enumerate(interpretable_variance['class_variances']):
                f.write(f"Class {i}: {var.item():.4f}\n")
            f.write("\nClass-wise Variances (Masked):\n")
            for i, var in enumerate(interpretable_variance['masked_class_variances']):
                f.write(f"Class {i}: {var.item():.4f}\n")
        
        # Save raw data as JSON for further analysis if needed
        with open(os.path.join(dataset_dir, 'feature_variance.json'), 'w') as f:
            json.dump(variance_results, f, indent=4)
    
    print("\nFeature variance analysis complete!")
    
    # Print results
    print("\nModel Comparison Results:")
    print("=" * 50)
    print("\nAccuracy (%):")
    print(f"Pretrained - Train: {metrics['pretrained_train_acc']:.2f}, Val: {metrics['pretrained_val_acc']:.2f}, Test: {metrics['pretrained_test_acc']:.2f}")
    print(f"Interpretable Unmasked - Train: {metrics['interpretable_unmasked_train_acc']:.2f}, Val: {metrics['interpretable_unmasked_val_acc']:.2f}, Test: {metrics['interpretable_unmasked_test_acc']:.2f}")
    print(f"Interpretable Masked - Train: {metrics['interpretable_masked_train_acc']:.2f}, Val: {metrics['interpretable_masked_val_acc']:.2f}, Test: {metrics['interpretable_masked_test_acc']:.2f}")

if __name__ == "__main__":
    main("cifar10_interpretable/cifar10_interpretable_20250502_173341") 