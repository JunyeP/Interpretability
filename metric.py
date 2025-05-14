import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Type, Any
import os
from tqdm import tqdm
import json
import warnings
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
from torchvision import transforms

# Suppress the specific warning
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

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
                           device: torch.device, num_classes: int, is_interpretable: bool = False) -> Dict[str, torch.Tensor]:
    """
    Analyze feature map variance across classes.
    
    Args:
        model: The model to analyze
        dataloader: DataLoader for the dataset
        device: Device to run computations on
        num_classes: Number of classes in the dataset
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
    
    # Initialize storage for feature maps
    feature_maps = {i: [] for i in range(num_classes)}
    masked_feature_maps = {i: [] for i in range(num_classes)} if is_interpretable else None
    
    print("Extracting feature maps...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Processing batches"):
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
    
    print("Computing class statistics...")
    # Calculate statistics for each class
    class_variances = torch.zeros(num_classes, device=device)
    class_means = torch.zeros(num_classes, features.shape[1], device=device)
    
    for i in tqdm(range(num_classes), desc="Computing class variances"):
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
        print("Computing masked feature statistics...")
        masked_class_variances = torch.zeros(num_classes, device=device)
        masked_class_means = torch.zeros(num_classes, features.shape[1], device=device)
        
        for i in tqdm(range(num_classes), desc="Computing masked class variances"):
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

def plot_feature_variance(variance_results: Dict[str, Any], save_path: str):
    """
    Create visualization for feature variance analysis.
    
    Args:
        variance_results: Dictionary containing variance analysis results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # 1. Class-wise variances comparison
    plt.subplot(1, 3, 1)
    classes = range(len(variance_results['pretrained']['class_variances']))
    pretrained_vars = variance_results['pretrained']['class_variances']
    interpretable_vars = variance_results['interpretable']['class_variances']
    interpretable_masked_vars = variance_results['interpretable']['masked_class_variances']
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, pretrained_vars, width, label='Pretrained')
    plt.bar(x, interpretable_vars, width, label='Interpretable')
    plt.bar(x + width, interpretable_masked_vars, width, label='Interpretable (Masked)')
    
    plt.xlabel('Class')
    plt.ylabel('Feature Variance')
    plt.title('Class-wise Feature Variance')
    plt.xticks(x, classes)
    plt.legend()
    
    # 2. Overall variance comparison
    plt.subplot(1, 3, 2)
    overall_vars = [
        variance_results['pretrained']['overall_variance'],
        variance_results['interpretable']['overall_variance'],
        variance_results['interpretable']['masked_overall_variance']
    ]
    plt.bar(['Pretrained', 'Interpretable', 'Interpretable (Masked)'], overall_vars)
    plt.ylabel('Overall Variance')
    plt.title('Overall Feature Variance')
    
    # 3. Feature value heatmap
    plt.subplot(1, 3, 3)
    feature_value = np.array(variance_results['pretrained']['class_means'])
    plt.imshow(feature_value, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Value')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Class')
    plt.title('Feature Value by Class')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_feature_space(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                         device: torch.device, num_classes: int, is_interpretable: bool = False) -> Dict[str, Any]:
    """
    Comprehensive feature space analysis including:
    1. PCA visualization
    2. Class separability metrics
    3. Feature consistency analysis
    """
    model.eval()
    features_by_class = {i: [] for i in range(num_classes)}
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for inputs, batch_labels in tqdm(dataloader, desc="Processing batches"):
            inputs = inputs.to(device)
            if is_interpretable:
                outputs = model(inputs)
                features = outputs['unmasked_mlp_features']
            else:
                features = model.get_features(inputs)
            
            # Store features by class
            for i in range(num_classes):
                class_mask = batch_labels == i
                if class_mask.any():
                    features_by_class[i].append(features[class_mask].cpu())
            labels_list.extend(batch_labels.cpu().numpy())
    
    print("Combining features...")
    # Combine features
    all_features = torch.cat([torch.cat(f, dim=0) for f in features_by_class.values() if f], dim=0).numpy()
    labels_np = np.array(labels_list)
    
    print("Performing PCA analysis...")
    # 1. PCA Analysis
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(all_features)
    
    # Create PCA visualization
    plt.figure(figsize=(10, 8))
    for i in tqdm(range(num_classes), desc="Creating PCA plot"):
        class_mask = labels_np == i
        plt.scatter(pca_features[class_mask, 0], 
                   pca_features[class_mask, 1],
                   label=f'Class {i}')
    plt.title('PCA Visualization of Feature Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    pca_plot = plt.gcf()
    
    print("Computing class separability metrics...")
    # 2. Class Separability Analysis
    lda = LinearDiscriminantAnalysis()
    lda_score = lda.fit_transform(all_features, labels_np)
    silhouette = silhouette_score(all_features, labels_np)
    davies_bouldin = davies_bouldin_score(all_features, labels_np)
    
    print("Analyzing feature consistency...")
    # 3. Feature Consistency Analysis
    feature_value = np.abs(all_features).mean(axis=0)
    feature_std = np.abs(all_features).std(axis=0)
    consistency_score = 1.0 / (1.0 + feature_std)
    
    # Create feature consistency visualization
    plt.figure(figsize=(15, 5))
    
    # Feature value distribution
    plt.subplot(1, 3, 1)
    plt.hist(feature_value, bins=50)
    plt.title('Feature Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    
    # Feature standard deviation distribution
    plt.subplot(1, 3, 2)
    plt.hist(feature_std, bins=50)
    plt.title('Feature Standard Deviation Distribution')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    
    # Consistency score distribution
    plt.subplot(1, 3, 3)
    plt.hist(consistency_score, bins=50)
    plt.title('Feature Consistency Score Distribution')
    plt.xlabel('Consistency Score')
    plt.ylabel('Count')
    
    plt.tight_layout()
    consistency_plot = plt.gcf()
    
    return {
        'pca': {
            'features': pca_features.tolist(),  # Convert to list for JSON serialization
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'plot': pca_plot
        },
        'separability': {
            'lda_score': lda_score.tolist(),
            'silhouette_score': float(silhouette),
            'davies_bouldin': float(davies_bouldin)
        },
        'consistency': {
            'feature_value': feature_value.tolist(),
            'feature_std': feature_std.tolist(),
            'consistency_score': consistency_score.tolist(),
            'plot': consistency_plot
        }
    }

def save_feature_variance_results(variance_results: Dict[str, Any], dataset_dir: str, dataset_name: str):
    """
    Save feature variance analysis results to both text and JSON files.
    
    Args:
        variance_results: Dictionary containing variance analysis results
        dataset_dir: Directory to save results
        dataset_name: Name of the dataset (train/val/test)
    """
    # Save to text file
    with open(os.path.join(dataset_dir, 'feature_variance.txt'), 'w') as f:
        f.write(f"Feature Variance Analysis for {dataset_name.upper()} set\n")
        f.write("=" * 50 + "\n\n")
        f.write("Pretrained Model:\n")
        f.write(f"Overall Variance: {variance_results['pretrained']['overall_variance']:.4f}\n")
        f.write("Class-wise Variances:\n")
        for i, var in enumerate(variance_results['pretrained']['class_variances']):
            f.write(f"Class {i}: {var:.4f}\n")
        f.write("\nInterpretable Model:\n")
        f.write(f"Overall Variance (Unmasked): {variance_results['interpretable']['overall_variance']:.4f}\n")
        f.write(f"Overall Variance (Masked): {variance_results['interpretable']['masked_overall_variance']:.4f}\n")
        f.write("Class-wise Variances (Unmasked):\n")
        for i, var in enumerate(variance_results['interpretable']['class_variances']):
            f.write(f"Class {i}: {var:.4f}\n")
        f.write("\nClass-wise Variances (Masked):\n")
        for i, var in enumerate(variance_results['interpretable']['masked_class_variances']):
            f.write(f"Class {i}: {var:.4f}\n")
    
    # Save raw data as JSON for further analysis
    with open(os.path.join(dataset_dir, 'feature_variance.json'), 'w') as f:
        json.dump(variance_results, f, indent=4)

def per_class_feature_stats(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, num_classes: int, is_interpretable: bool = False):
    """
    Compute per-class feature standard deviation and consistency for each feature.
    Returns:
        - per_class_std: shape (num_classes, num_features)
        - per_class_consistency: shape (num_classes, num_features)
    """
    model.eval()
    features_by_class = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Per-class feature stats"):
            inputs = inputs.to(device)
            if is_interpretable:
                outputs = model(inputs)
                features = outputs['unmasked_mlp_features']
            else:
                features = model.get_features(inputs)
            for i in range(num_classes):
                class_mask = labels == i
                if class_mask.any():
                    features_by_class[i].append(features[class_mask].cpu())
    per_class_std = []
    per_class_consistency = []
    for i in range(num_classes):
        if features_by_class[i]:
            class_features = torch.cat(features_by_class[i], dim=0).numpy()
            std = np.std(class_features, axis=0)
            consistency = 1.0 / (1.0 + std)
            per_class_std.append(std)
            per_class_consistency.append(consistency)
        else:
            per_class_std.append(np.zeros(class_features.shape[1]))
            per_class_consistency.append(np.ones(class_features.shape[1]))
    return np.array(per_class_std), np.array(per_class_consistency)

def plot_per_class_heatmap(per_class_matrix, save_path, title, xlabel='Feature', ylabel='Class', vmin=None, vmax=None):
    plt.figure(figsize=(12, 6))
    sns.heatmap(per_class_matrix, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_feature_overlap(per_class_matrix, top_k=10):
    """
    Compute mean pairwise overlap (Jaccard index) of top-k features between all class pairs.
    Returns:
        - overlap_matrix: shape (num_classes, num_classes)
        - mean_overlap: scalar
    """
    num_classes, num_features = per_class_matrix.shape
    topk_sets = [set(np.argsort(row)[-top_k:]) for row in per_class_matrix]
    overlap_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                inter = len(topk_sets[i] & topk_sets[j])
                union = len(topk_sets[i] | topk_sets[j])
                overlap_matrix[i, j] = inter / union if union > 0 else 0.0
    mean_overlap = (np.sum(overlap_matrix) - num_classes) / (num_classes * (num_classes - 1))
    return overlap_matrix, mean_overlap

def plot_overlap_matrix(overlap_matrix, save_path, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_mean_per_class_stats(per_class_std, per_class_consistency, save_path, model_name):
    """
    Plot the mean per-class std and mean per-class consistency for each feature as a single PNG.
    """
    mean_std = per_class_std.mean(axis=0)
    mean_consistency = per_class_consistency.mean(axis=0)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mean_std)
    plt.title(f'{model_name.capitalize()} Mean Per-Class Feature Std')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Std')
    plt.subplot(1, 2, 2)
    plt.plot(mean_consistency)
    plt.title(f'{model_name.capitalize()} Mean Per-Class Feature Consistency')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Consistency')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_shap_visualization(pretrained_model, interpretable_model, data_loader, device, save_dir):
    from shap_visualizer import SHAPVisualizer
    visualizer = SHAPVisualizer(pretrained_model, interpretable_model, data_loader, device, num_samples=10000, num_visualize=10)
    visualizer.run(save_dir)

def evaluate_and_plot_models(pretrained_model, interpretable_model, train_loader, val_loader, test_loader, device, criterion, metrics_dir, num_classes: int):
    metrics = {}
    # Pretrained model evaluation
    metrics['pretrained_train_acc'] = evaluate_accuracy(pretrained_model, train_loader, device)
    metrics['pretrained_val_acc'] = evaluate_accuracy(pretrained_model, val_loader, device)
    metrics['pretrained_test_acc'] = evaluate_accuracy(pretrained_model, test_loader, device)
    metrics['pretrained_train_loss'] = evaluate_loss(pretrained_model, train_loader, criterion, device)
    metrics['pretrained_val_loss'] = evaluate_loss(pretrained_model, val_loader, criterion, device)
    metrics['pretrained_test_loss'] = evaluate_loss(pretrained_model, test_loader, criterion, device)
    # Interpretable model evaluation (unmasked)
    metrics['interpretable_unmasked_train_acc'] = evaluate_accuracy(interpretable_model, train_loader, device, is_interpretable=True)
    metrics['interpretable_unmasked_val_acc'] = evaluate_accuracy(interpretable_model, val_loader, device, is_interpretable=True)
    metrics['interpretable_unmasked_test_acc'] = evaluate_accuracy(interpretable_model, test_loader, device, is_interpretable=True)
    metrics['interpretable_unmasked_train_loss'] = evaluate_loss(interpretable_model, train_loader, criterion, device, is_interpretable=True)
    metrics['interpretable_unmasked_val_loss'] = evaluate_loss(interpretable_model, val_loader, criterion, device, is_interpretable=True)
    metrics['interpretable_unmasked_test_loss'] = evaluate_loss(interpretable_model, test_loader, criterion, device, is_interpretable=True)
    # Interpretable model evaluation (masked)
    metrics['interpretable_masked_train_acc'] = evaluate_accuracy(interpretable_model, train_loader, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_val_acc'] = evaluate_accuracy(interpretable_model, val_loader, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_test_acc'] = evaluate_accuracy(interpretable_model, test_loader, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_train_loss'] = evaluate_loss(interpretable_model, train_loader, criterion, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_val_loss'] = evaluate_loss(interpretable_model, val_loader, criterion, device, is_interpretable=True, masked_path=True)
    metrics['interpretable_masked_test_loss'] = evaluate_loss(interpretable_model, test_loader, criterion, device, is_interpretable=True, masked_path=True)
    # AUROC curves
    roc_curves = {}
    roc_curves['Pretrained (Train)'] = compute_auroc(pretrained_model, train_loader, device, num_classes=num_classes)
    roc_curves['Pretrained (Val)'] = compute_auroc(pretrained_model, val_loader, device, num_classes=num_classes)
    roc_curves['Pretrained (Test)'] = compute_auroc(pretrained_model, test_loader, device, num_classes=num_classes)
    roc_curves['Interpretable Unmasked (Train)'] = compute_auroc(interpretable_model, train_loader, device, num_classes=num_classes, is_interpretable=True)
    roc_curves['Interpretable Unmasked (Val)'] = compute_auroc(interpretable_model, val_loader, device, num_classes=num_classes, is_interpretable=True)
    roc_curves['Interpretable Unmasked (Test)'] = compute_auroc(interpretable_model, test_loader, device, num_classes=num_classes, is_interpretable=True)
    roc_curves['Interpretable Masked (Train)'] = compute_auroc(interpretable_model, train_loader, device, num_classes=num_classes, is_interpretable=True, masked_path=True)
    roc_curves['Interpretable Masked (Val)'] = compute_auroc(interpretable_model, val_loader, device, num_classes=num_classes, is_interpretable=True, masked_path=True)
    roc_curves['Interpretable Masked (Test)'] = compute_auroc(interpretable_model, test_loader, device, num_classes=num_classes, is_interpretable=True, masked_path=True)
    # Plot metrics
    plot_metrics(metrics, os.path.join(metrics_dir, 'metrics_comparison.png'))
    plot_auroc_curves(roc_curves, os.path.join(metrics_dir, 'auroc_curves.png'))
    return metrics, roc_curves

def run_feature_analysis_for_split(pretrained_model, interpretable_model, data_loader, device, dataset_name, dataset_dir, num_classes: int):
    """Run feature analysis for a specific dataset split."""
    # Ensure dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 1. Feature Variance Analysis
    pretrained_variance = analyze_feature_variance(pretrained_model, data_loader, device, num_classes=num_classes)
    interpretable_variance = analyze_feature_variance(interpretable_model, data_loader, device, num_classes=num_classes, is_interpretable=True)
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
    
    # Save feature variance plot
    plot_feature_variance(variance_results, os.path.join(dataset_dir, 'feature_variance.png'))
    
    # 2. Feature Space Analysis
    pretrained_analysis = analyze_feature_space(pretrained_model, data_loader, device, num_classes=num_classes)
    pretrained_analysis['pca']['plot'].savefig(os.path.join(dataset_dir, 'pretrained_pca.png'))
    pretrained_analysis['consistency']['plot'].savefig(os.path.join(dataset_dir, 'pretrained_consistency.png'))
    plt.close('all')
    
    interpretable_analysis = analyze_feature_space(interpretable_model, data_loader, device, num_classes=num_classes, is_interpretable=True)
    interpretable_analysis['pca']['plot'].savefig(os.path.join(dataset_dir, 'interpretable_pca.png'))
    interpretable_analysis['consistency']['plot'].savefig(os.path.join(dataset_dir, 'interpretable_consistency.png'))
    plt.close('all')
    
    # Per-class feature std/consistency and overlap analysis
    for model, model_name in [(pretrained_model, 'pretrained'), (interpretable_model, 'interpretable')]:
        per_class_std, per_class_consistency = per_class_feature_stats(model, data_loader, device, num_classes=num_classes, is_interpretable=(model_name=='interpretable'))
        
        # Save per-class heatmaps
        plot_per_class_heatmap(per_class_std, os.path.join(dataset_dir, f'{model_name}_per_class_std.png'), 
                             f'{model_name.capitalize()} Per-Class Feature Std')
        plot_per_class_heatmap(per_class_consistency, os.path.join(dataset_dir, f'{model_name}_per_class_consistency.png'), 
                             f'{model_name.capitalize()} Per-Class Feature Consistency', vmin=0, vmax=1)
        
        # Save mean per-class stats
        plot_mean_per_class_stats(per_class_std, per_class_consistency, 
                                os.path.join(dataset_dir, f'{model_name}_mean_per_class_stats.png'), 
                                model_name)
        
        # Compute and save overlap analysis
        overlap_matrix, mean_overlap = compute_feature_overlap(per_class_std, top_k=10)
        plot_overlap_matrix(overlap_matrix, 
                          os.path.join(dataset_dir, f'{model_name}_feature_overlap.png'), 
                          f'{model_name.capitalize()} Feature Overlap (Top-10)')
        
        # Save mean overlap to text file
        with open(os.path.join(dataset_dir, f'{model_name}_mean_overlap.txt'), 'w') as f:
            f.write(f'Mean pairwise overlap (top-10 features): {mean_overlap:.4f}\n')
    
    # Visualization for separability scores
    sep_scores = {
        'Model': ['Pretrained', 'Interpretable'],
        'Silhouette': [pretrained_analysis['separability']['silhouette_score'], 
                      interpretable_analysis['separability']['silhouette_score']],
        'Davies-Bouldin': [pretrained_analysis['separability']['davies_bouldin'], 
                          interpretable_analysis['separability']['davies_bouldin']]
    }
    
    # Plot and save silhouette scores
    plt.figure(figsize=(6, 5))
    x = np.arange(len(sep_scores['Model']))
    width = 0.5
    bars1 = plt.bar(x, sep_scores['Silhouette'], width, color='tab:blue', 
                   label='Silhouette Score (higher is better)')
    plt.xticks(x, sep_scores['Model'])
    plt.ylabel('Silhouette Score')
    plt.title('Feature Separability: Silhouette Score (higher is better)')
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, 'feature_separability_silhouette.png'))
    plt.close()
    
    # Plot and save Davies-Bouldin scores
    plt.figure(figsize=(6, 5))
    bars2 = plt.bar(x, sep_scores['Davies-Bouldin'], width, color='tab:orange', 
                   label='Davies-Bouldin (lower is better)')
    plt.xticks(x, sep_scores['Model'])
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Feature Separability: Davies-Bouldin Index (lower is better)')
    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, 'feature_separability_davies_bouldin.png'))
    plt.close()

def pgd_attack(model, inputs, labels, epsilon, device, num_steps=10, step_size=0.01,
               is_interpretable=False, masked_path=False):
    """Generate PGD adversarial examples."""
    adv_inputs = inputs.clone().detach()
    
    for step in range(num_steps):
        adv_inputs.requires_grad = True
        
        # Forward pass
        outputs = model(adv_inputs)
        print(f"\nStep {step + 1}/{num_steps}:")
        print(f"Model output type: {type(outputs)}")
        
        # Handle model output
        if isinstance(outputs, dict):
            print(f"Output dictionary keys: {outputs.keys()}")
            if is_interpretable:
                logits = outputs['masked_logits'] if masked_path else outputs['unmasked_logits']
            else:
                # For non-interpretable model, use unmasked_logits
                logits = outputs['unmasked_logits']
        else:
            logits = outputs
            
        print(f"Logits type: {type(logits)}")
        print(f"Logits shape: {logits.shape}")
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        print(f"Labels shape: {labels.shape}")
        print(f"Labels type: {type(labels)}")
        loss = criterion(logits, labels)
        print(f"Loss value: {loss.item()}")
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial examples
        with torch.no_grad():
            adv_inputs = adv_inputs + step_size * torch.sign(adv_inputs.grad.data)
            delta = torch.clamp(adv_inputs - inputs, -epsilon, epsilon)
            adv_inputs = torch.clamp(inputs + delta, 0, 1)
    
    return adv_inputs

def fgsm_attack(model, inputs, labels, epsilon, device, is_interpretable=False, masked_path=False):
    """Generate FGSM adversarial examples."""
    inputs.requires_grad = True
    
    # Forward pass
    outputs = model(inputs)
    
    # Handle interpretable model output
    if is_interpretable:
        if not isinstance(outputs, dict):
            raise TypeError("Interpretable model should return a dictionary")
        logits = outputs['masked_logits'] if masked_path else outputs['unmasked_logits']
    else:
        logits = outputs
    
    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial examples
    with torch.no_grad():
        perturbation = epsilon * torch.sign(inputs.grad.data)
        adv_inputs = inputs + perturbation
        adv_inputs = torch.clamp(adv_inputs, 0, 1)
    
    return adv_inputs

def evaluate_adversarial_robustness(model, dataloader, device, epsilon_range, attack_type='fgsm', 
                                  is_interpretable=False, masked_path=False):
    """
    Evaluate model robustness against adversarial attacks.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        epsilon_range: List of epsilon values to test
        attack_type: Type of attack ('fgsm' or 'pgd')
        is_interpretable: Whether model is interpretable
        masked_path: Whether to use masked path for interpretable model
        
    Returns:
        Dictionary containing accuracy results for each epsilon
    """
    results = {eps: {'accuracy': 0.0, 'total': 0} for eps in epsilon_range}
    
    for eps in epsilon_range:
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {attack_type.upper()} (ε={eps})"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Generate adversarial examples
            if attack_type == 'fgsm':
                adv_inputs = fgsm_attack(model, inputs, labels, eps, device, 
                                       is_interpretable, masked_path)
            else:  # pgd
                adv_inputs = pgd_attack(model, inputs, labels, eps, device,
                                      is_interpretable, masked_path)
            
            # Evaluate on adversarial examples
            with torch.no_grad():
                if is_interpretable:
                    outputs = model(adv_inputs)
                    if isinstance(outputs, dict):
                        logits = outputs['masked_logits'] if masked_path else outputs['unmasked_logits']
                    else:
                        logits = outputs
                else:
                    logits = model(adv_inputs)
                
                # Ensure logits is a tensor
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(f"Expected logits to be a tensor, got {type(logits)}")
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        results[eps]['accuracy'] = 100 * correct / total
        results[eps]['total'] = total
    
    return results

def evaluate_augmentation_robustness(model, dataloader, device, augmentation_types,
                                   is_interpretable=False, masked_path=False):
    """
    Evaluate model robustness against various augmentations.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        augmentation_types: List of augmentation types to test
        is_interpretable: Whether model is interpretable
        masked_path: Whether to use masked path for interpretable model
        
    Returns:
        Dictionary containing accuracy results for each augmentation type
    """
    results = {aug_type: {'accuracy': 0.0, 'total': 0} for aug_type in augmentation_types}
    
    for aug_type in augmentation_types:
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {aug_type}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply augmentation
            aug_inputs = apply_augmentation(inputs, aug_type)
            
            # Evaluate on augmented examples
            with torch.no_grad():
                if is_interpretable:
                    outputs = model(aug_inputs)
                    logits = outputs['masked_logits'] if masked_path else outputs['unmasked_logits']
                else:
                    logits = model(aug_inputs)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        results[aug_type]['accuracy'] = 100 * correct / total
        results[aug_type]['total'] = total
    
    return results

def apply_augmentation(inputs, aug_type):
    """Apply specified augmentation to input tensor."""
    if aug_type == 'color_jitter':
        # Random color jittering
        brightness = 0.5
        contrast = 0.5
        saturation = 0.5
        hue = 0.1
        return transforms.ColorJitter(brightness, contrast, saturation, hue)(inputs)
    
    elif aug_type == 'gaussian_noise':
        # Add Gaussian noise
        noise = torch.randn_like(inputs) * 0.1
        return torch.clamp(inputs + noise, 0, 1)
    
    elif aug_type == 'random_erase':
        # Random erasing
        return transforms.RandomErasing(p=1.0)(inputs)
    
    elif aug_type == 'rotation':
        # Random rotation
        return transforms.RandomRotation(30)(inputs)
    
    elif aug_type == 'scale':
        # Random scaling
        return transforms.RandomAffine(0, scale=(0.8, 1.2))(inputs)
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def plot_adversarial_robustness(results, save_path):
    """Plot adversarial robustness results."""
    plt.figure(figsize=(10, 6))
    
    for model_name, model_results in results.items():
        epsilons = list(model_results.keys())
        accuracies = [model_results[eps]['accuracy'] for eps in epsilons]
        plt.plot(epsilons, accuracies, marker='o', label=model_name)
    
    plt.xlabel('Epsilon (Perturbation Magnitude)')
    plt.ylabel('Accuracy (%)')
    plt.title('Adversarial Robustness')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_augmentation_robustness(results, save_path):
    """Plot augmentation robustness results."""
    plt.figure(figsize=(12, 6))
    
    models = list(results.keys())
    aug_types = list(results[models[0]].keys())
    x = np.arange(len(aug_types))
    width = 0.8 / len(models)
    
    for i, model_name in enumerate(models):
        accuracies = [results[model_name][aug]['accuracy'] for aug in aug_types]
        plt.bar(x + i * width, accuracies, width, label=model_name)
    
    plt.xlabel('Augmentation Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Augmentation Robustness')
    plt.xticks(x + width * (len(models) - 1) / 2, aug_types, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_robustness(pretrained_model, interpretable_model, train_loader, val_loader, test_loader, 
                       device, metrics_dir, num_classes: int):
    """
    Evaluate model robustness against adversarial attacks and augmentations on all datasets.
    
    Args:
        pretrained_model: Pretrained model
        interpretable_model: Interpretable model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run evaluation on
        metrics_dir: Directory to save results
        num_classes: Number of classes in the dataset
    """
    # Create directories for results
    adversarial_dir = os.path.join(metrics_dir, 'adversarial_robustness')
    augmentation_dir = os.path.join(metrics_dir, 'augmentation_robustness')
    os.makedirs(adversarial_dir, exist_ok=True)
    os.makedirs(augmentation_dir, exist_ok=True)
    
    # Parameters for evaluation
    epsilon_range = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    attack_type = 'fgsm'  # Only use FGSM attack
    augmentation_types = [
        'color_jitter',
        'gaussian_noise',
        'random_erase',
        'rotation',
        'scale'
    ]
    
    # Evaluate on each dataset
    for dataset_name, data_loader in [
        ('train', train_loader),
        ('val', val_loader),
        ('test', test_loader)
    ]:
        print(f"\nEvaluating robustness on {dataset_name} set...")
        
        # Create dataset-specific directories
        dataset_adversarial_dir = os.path.join(adversarial_dir, dataset_name)
        dataset_augmentation_dir = os.path.join(augmentation_dir, dataset_name)
        os.makedirs(dataset_adversarial_dir, exist_ok=True)
        os.makedirs(dataset_augmentation_dir, exist_ok=True)
        
        # 1. Adversarial Robustness Evaluation
        print(f"\nRunning {attack_type.upper()} attack on {dataset_name} set...")
        
        adversarial_results = {}
        
        # Evaluate pretrained model
        adversarial_results['pretrained'] = evaluate_adversarial_robustness(
            pretrained_model, data_loader, device, epsilon_range, attack_type
        )
        
        # Evaluate interpretable model (unmasked)
        adversarial_results['interpretable_unmasked'] = evaluate_adversarial_robustness(
            interpretable_model, data_loader, device, epsilon_range, attack_type,
            is_interpretable=True
        )
        
        # Evaluate interpretable model (masked)
        adversarial_results['interpretable_masked'] = evaluate_adversarial_robustness(
            interpretable_model, data_loader, device, epsilon_range, attack_type,
            is_interpretable=True, masked_path=True
        )
        
        # Plot results
        plot_adversarial_robustness(
            adversarial_results,
            os.path.join(dataset_adversarial_dir, f'{attack_type}_robustness.png')
        )
        
        # Save raw results
        with open(os.path.join(dataset_adversarial_dir, f'{attack_type}_results.json'), 'w') as f:
            json.dump(adversarial_results, f, indent=4)
        
        # 2. Augmentation Robustness Evaluation
        print(f"\nEvaluating augmentation robustness on {dataset_name} set...")
        
        augmentation_results = {}
        
        # Evaluate pretrained model
        augmentation_results['pretrained'] = evaluate_augmentation_robustness(
            pretrained_model, data_loader, device, augmentation_types
        )
        
        # Evaluate interpretable model (unmasked)
        augmentation_results['interpretable_unmasked'] = evaluate_augmentation_robustness(
            interpretable_model, data_loader, device, augmentation_types,
            is_interpretable=True
        )
        
        # Evaluate interpretable model (masked)
        augmentation_results['interpretable_masked'] = evaluate_augmentation_robustness(
            interpretable_model, data_loader, device, augmentation_types,
            is_interpretable=True, masked_path=True
        )
        
        # Plot results
        plot_augmentation_robustness(
            augmentation_results,
            os.path.join(dataset_augmentation_dir, 'augmentation_robustness.png')
        )
        
        # Save raw results
        with open(os.path.join(dataset_augmentation_dir, 'augmentation_results.json'), 'w') as f:
            json.dump(augmentation_results, f, indent=4)
