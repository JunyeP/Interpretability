import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Type
import os
from tqdm import tqdm

def load_models(pretrained_path: str, interpretable_path: str, device: torch.device,
               pretrained_model_class: Optional[Type[nn.Module]] = None,
               interpretable_model_class: Optional[Type[nn.Module]] = None) -> Tuple[nn.Module, nn.Module]:
    """
    Load pretrained model and interpretable model from checkpoints.
    Handles both full models and state dictionaries.
    
    Args:
        pretrained_path: Path to pretrained model checkpoint
        interpretable_path: Path to interpretable model checkpoint
        device: Device to load models on
        pretrained_model_class: Class of the pretrained model (needed if loading state dict)
        interpretable_model_class: Class of the interpretable model (needed if loading state dict)
        
    Returns:
        Tuple of (pretrained_model, interpretable_model)
    """
    # Load pretrained model
    pretrained_state = torch.load(pretrained_path, map_location=device)
    if isinstance(pretrained_state, dict):
        if pretrained_model_class is None:
            raise ValueError("pretrained_model_class must be provided when loading a state dict")
        pretrained_model = pretrained_model_class()
        pretrained_model.load_state_dict(pretrained_state)
    else:
        pretrained_model = pretrained_state
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    
    # Load interpretable model
    interpretable_state = torch.load(interpretable_path, map_location=device)
    if isinstance(interpretable_state, dict):
        if interpretable_model_class is None:
            raise ValueError("interpretable_model_class must be provided when loading a state dict")
        interpretable_model = interpretable_model_class()
        interpretable_model.load_state_dict(interpretable_state)
    else:
        interpretable_model = interpretable_state
    interpretable_model = interpretable_model.to(device)
    interpretable_model.eval()
    
    return pretrained_model, interpretable_model

def evaluate_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                    device: torch.device, is_interpretable: bool = False, masked_path: bool = False) -> float:
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        is_interpretable: Whether model is interpretable (affects forward pass)
        masked_path: Whether to use masked_logits instead of unmasked_logits
        
    Returns:
        Accuracy as float
    """
    correct = 0
    total = 0
    
    # Verify model is in eval mode
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating accuracy"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_interpretable:
                # Get full output dictionary from interpretable model
                outputs_dict = model(inputs)
                
                # Verify output dictionary contains required keys
                required_keys = ['unmasked_logits', 'masked_logits', 'mask', 'radial_mask', 'binary_mask']
                for key in required_keys:
                    if key not in outputs_dict:
                        raise ValueError(f"Interpretable model output missing required key: {key}")
                
                # Select appropriate logits based on path
                outputs = outputs_dict['masked_logits'] if masked_path else outputs_dict['unmasked_logits']
                
                # Verify mask shapes and values
                mask = outputs_dict['mask']
                radial_mask = outputs_dict['radial_mask']
                binary_mask = outputs_dict['binary_mask']
                
                if mask.min() < 0 or mask.max() > 1:
                    print(f"Warning: Soft mask values outside [0,1] range: min={mask.min()}, max={mask.max()}")
                if radial_mask.min() < 0 or radial_mask.max() > 1:
                    print(f"Warning: Radial mask values outside [0,1] range: min={radial_mask.min()}, max={radial_mask.max()}")
                if not torch.all((binary_mask == 0) | (binary_mask == 1)):
                    print("Warning: Binary mask contains values other than 0 and 1")
            else:
                # For pretrained model, directly get logits
                outputs = model(inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

def evaluate_loss(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                 criterion: nn.Module, device: torch.device,
                 is_interpretable: bool = False, masked_path: bool = False) -> float:
    """
    Evaluate model loss on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        is_interpretable: Whether model is interpretable
        masked_path: Whether to use masked_logits instead of unmasked_logits
        
    Returns:
        Average loss as float
    """
    total_loss = 0
    total_samples = 0
    
    # Verify model is in eval mode
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating loss"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_interpretable:
                # Get full output dictionary from interpretable model
                outputs_dict = model(inputs)
                
                # Verify output dictionary contains required keys
                required_keys = ['unmasked_logits', 'masked_logits', 'mask', 'radial_mask', 'binary_mask']
                for key in required_keys:
                    if key not in outputs_dict:
                        raise ValueError(f"Interpretable model output missing required key: {key}")
                
                # Select appropriate logits based on path
                outputs = outputs_dict['masked_logits'] if masked_path else outputs_dict['unmasked_logits']
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
    return total_loss / total_samples

def compute_auroc(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                 device: torch.device, num_classes: int,
                 is_interpretable: bool = False, masked_path: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute AUROC curves for each class and average.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        is_interpretable: Whether model is interpretable
        masked_path: Whether to use masked_logits instead of unmasked_logits
        
    Returns:
        Tuple of (fpr, tpr) arrays for average ROC curve
    """
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Computing AUROC"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_interpretable:
                outputs_dict = model(inputs)
                outputs = outputs_dict['masked_logits'] if masked_path else outputs_dict['unmasked_logits']
            else:
                outputs = model(inputs)
                
            probs = torch.softmax(outputs, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    
    return all_fpr, mean_tpr

def plot_metrics(metrics: Dict[str, float], save_path: str):
    """
    Plot and save metrics comparison.
    
    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save plot
    """
    # Create two separate figures for accuracy and loss
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy metrics
    plt.subplot(1, 2, 1)
    models = ['Pretrained', 'Interpretable (Unmasked)', 'Interpretable (Masked)']
    train_acc = [metrics['pretrained_train_acc'], 
                metrics['interpretable_unmasked_train_acc'],
                metrics['interpretable_masked_train_acc']]
    val_acc = [metrics['pretrained_val_acc'],
              metrics['interpretable_unmasked_val_acc'],
              metrics['interpretable_masked_val_acc']]
    test_acc = [metrics['pretrained_test_acc'],
               metrics['interpretable_unmasked_test_acc'],
               metrics['interpretable_masked_test_acc']]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, train_acc, width, label='Train')
    plt.bar(x, val_acc, width, label='Val')
    plt.bar(x + width, test_acc, width, label='Test')
    plt.xticks(x, models)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.legend()
    
    # Plot loss metrics
    plt.subplot(1, 2, 2)
    train_loss = [metrics['pretrained_train_loss'],
                 metrics['interpretable_unmasked_train_loss'],
                 metrics['interpretable_masked_train_loss']]
    val_loss = [metrics['pretrained_val_loss'],
               metrics['interpretable_unmasked_val_loss'],
               metrics['interpretable_masked_val_loss']]
    test_loss = [metrics['pretrained_test_loss'],
                metrics['interpretable_unmasked_test_loss'],
                metrics['interpretable_masked_test_loss']]
    
    plt.bar(x - width, train_loss, width, label='Train')
    plt.bar(x, val_loss, width, label='Val')
    plt.bar(x + width, test_loss, width, label='Test')
    plt.xticks(x, models)
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_auroc_curves(roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str):
    """
    Plot and save AUROC curves.
    
    Args:
        roc_curves: Dictionary of model names and their ROC curves
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors and line styles for different models
    colors = {
        'Pretrained': 'blue',
        'Interpretable Unmasked': 'green',
        'Interpretable Masked': 'red'
    }
    
    # Plot each curve with appropriate styling
    for model_name, (fpr, tpr) in roc_curves.items():
        base_name = model_name.split(' (')[0]  # Get base model name
        is_train = '(Train)' in model_name
        line_style = '-' if is_train else '--'  # Solid for train, dashed for val
        
        plt.plot(fpr, tpr, 
                label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})',
                color=colors[base_name],
                linestyle=line_style,
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison (Train vs Validation)')
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    plt.close()

def analyze_shap_extremes(shap_values: torch.Tensor, upper_threshold: float = 0.8, lower_threshold: float = 0.2) -> Dict[str, float]:
    """
    Analyze SHAP values by counting extreme values (important and non-important features).
    
    Args:
        shap_values: SHAP values tensor of shape (batch_size, channels, height, width)
        upper_threshold: Upper threshold for important features (default: 0.8)
        lower_threshold: Lower threshold for non-important features (default: 0.2)
    
    Returns:
        Dictionary containing:
        - important_count: Number of features above upper_threshold
        - non_important_count: Number of features below lower_threshold
        - important_ratio: Ratio of important features to total features
        - non_important_ratio: Ratio of non-important features to total features
    """
    # Normalize SHAP values to [0, 1]
    shap_norm = (shap_values - shap_values.min()) / (shap_values.max() - shap_values.min() + 1e-10)
    
    # Count important and non-important features
    important_mask = shap_norm > upper_threshold
    non_important_mask = shap_norm < lower_threshold
    
    total_features = shap_values.numel()
    important_count = important_mask.sum().item()
    non_important_count = non_important_mask.sum().item()
    
    return {
        'important_count': important_count,
        'non_important_count': non_important_count,
        'important_ratio': important_count / total_features,
        'non_important_ratio': non_important_count / total_features
    }

def analyze_feature_variance(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                           device: torch.device, is_interpretable: bool = False) -> Dict[str, torch.Tensor]:
    """
    Analyze feature map variance across classes.
    
    Args:
        model: The model to analyze
        dataloader: DataLoader for the dataset
        device: Device to run computations on
        is_interpretable: Whether the model is interpretable
    
    Returns:
        Dictionary containing:
        - class_variances: Variance of feature maps for each class
        - class_means: Mean feature maps for each class
        - overall_variance: Overall variance across all classes
        - masked_class_variances: (if interpretable) Variance of masked features for each class
        - masked_class_means: (if interpretable) Mean masked features for each class
        - masked_overall_variance: (if interpretable) Overall variance of masked features
    """
    model.eval()
    num_classes = 10  # CIFAR-10 specific
    
    # Initialize storage for feature maps
    feature_maps = {i: [] for i in range(num_classes)}
    masked_feature_maps = {i: [] for i in range(num_classes)} if is_interpretable else None
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            if is_interpretable:
                # Get feature maps from interpretable model
                outputs = model(inputs)
                features = outputs['unmasked_mlp_features']  # Get feature maps from MLP
                masked_features = outputs['masked_mlp_features']  # Get masked feature maps
            else:
                # Get feature maps from standard model using get_features()
                features = model.get_features(inputs)  # Get feature maps from MLP
                masked_features = None
            
            # Store feature maps by class
            for i in range(num_classes):
                class_mask = labels == i
                if class_mask.any():
                    class_features = features[class_mask]
                    feature_maps[i].append(class_features)
                    
                    if is_interpretable:
                        class_masked_features = masked_features[class_mask]
                        masked_feature_maps[i].append(class_masked_features)
    
    # Calculate statistics for each class
    class_variances = torch.zeros(num_classes, device=device)
    class_means = torch.zeros(num_classes, features.shape[1], device=device)
    
    for i in range(num_classes):
        if feature_maps[i]:
            class_features = torch.cat(feature_maps[i], dim=0)
            class_means[i] = class_features.mean(dim=0)
            class_variances[i] = class_features.var(dim=0).mean()
    
    # Calculate overall variance
    overall_variance = class_variances.mean()
    
    result = {
        'class_variances': class_variances,
        'class_means': class_means,
        'overall_variance': overall_variance
    }
    
    # Calculate masked feature statistics if interpretable model
    if is_interpretable:
        masked_class_variances = torch.zeros(num_classes, device=device)
        masked_class_means = torch.zeros(num_classes, features.shape[1], device=device)
        
        for i in range(num_classes):
            if masked_feature_maps[i]:
                class_masked_features = torch.cat(masked_feature_maps[i], dim=0)
                masked_class_means[i] = class_masked_features.mean(dim=0)
                masked_class_variances[i] = class_masked_features.var(dim=0).mean()
        
        masked_overall_variance = masked_class_variances.mean()
        
        result.update({
            'masked_class_variances': masked_class_variances,
            'masked_class_means': masked_class_means,
            'masked_overall_variance': masked_overall_variance
        })
    
    return result
