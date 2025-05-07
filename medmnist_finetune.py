import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import warnings
from PIL import Image
from torchvision.transforms import InterpolationMode

# Suppress specific NetworkX warning
warnings.filterwarnings("ignore", message="networkx backend defined more than once")

import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
from torchvision.models import resnet18
from interpretable_model import InterpretableModel
from toolkits import (
    MaskedLoss,
    get_accuracy,
    validate,
    visualize_results,
    plot_training_metrics,
    evaluate_model,
    train_model,
    setup_experiment,
    load_pretrained_weights,
    perform_initial_evaluation
)

def load_config(config_path=None):
    """Load configuration by merging user-provided file over default parameters"""
    # Default configuration with all required keys
    default_config = {
        'data_flag': 'pathmnist',
        'batch_size': 64,
        'num_workers': 2,
        'num_epochs': 5,
        'learning_rate': 0.0001,
        'experiment_name': 'medmnist_finetune_experiment',
        'log_dir': './logs',
        'pretrained_path': './pretrain/pathmnist/medmnist_resnet18/best_model.pth',
        'freeze_cnn': False,
        'lambda_mask': 1.0,
        'lambda_fully_masked': 0.2,
        'lambda_alignment': 0.5,
        'upper_mask_level_threshold': 0.8,
        'dynamic_masked_weight_min': 1.0,
        'dynamic_masked_weight_max': 5.0,
        'radial_radius': 3,
        'radial_decay': 0.5,
        'enable_radial_mask_noise': False,
        'radial_mask_noise_range': 0.1,
        # Scheduling parameters
        'start_epoch_mask': 0,
        'start_epoch_fully_masked': 0,
        'start_epoch_alignment': 0,
        'initial_lambda_mask': 1.0,
        'final_lambda_mask': 1.0,
        'initial_lambda_fully_masked': 0.2,
        'final_lambda_fully_masked': 0.2,
        'initial_lambda_alignment': 0.5,
        'final_lambda_alignment': 0.5,
        # Entropy regularization scheduling parameters
        'start_epoch_entropy': 0,
        'initial_lambda_entropy': 0.0,
        'final_lambda_entropy': 0.0,
        # Visualization and checkpoint intervals
        'visualization_interval': 1,
        'checkpoint_interval': 1,
    }
    # If a config path is provided, load and merge
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    return default_config

def load_medmnist_dataset(config):
    data_flag = config['data_flag']
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']
    img_size = 224
    
    # Build train transforms: include augmentation if enabled
    train_transform_list = []
    if config.get('enable_augmentation', False):
        aug_cfg = config.get('augmentation_transforms', {})
        # Random or resized crop
        if aug_cfg.get('random_crop', False):
            train_transform_list.append(
                transforms.RandomResizedCrop(
                    img_size,
                    interpolation=InterpolationMode.BILINEAR
                )
            )
        else:
            train_transform_list.append(
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True
                )
            )
        if aug_cfg.get('random_horizontal_flip', False):
            train_transform_list.append(transforms.RandomHorizontalFlip())
        if aug_cfg.get('random_rotation', 0):
            train_transform_list.append(transforms.RandomRotation(aug_cfg['random_rotation']))
        if aug_cfg.get('color_jitter', {}):
            train_transform_list.append(transforms.ColorJitter(**aug_cfg['color_jitter']))
    else:
        train_transform_list.append(
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            )
        )
    train_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5]*n_channels, std=[.5]*n_channels)
    ]
    train_transform = transforms.Compose(train_transform_list)
    # Validation and test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5]*n_channels, std=[.5]*n_channels)
    ])

    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset = DataClass(split='val', transform=val_transform, download=True)
    test_dataset = DataClass(split='test', transform=val_transform, download=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    return train_loader, val_loader, test_loader, info

class MaskGenerator(nn.Module):
    def __init__(self, in_channels=3, base_filters=64):
        super(MaskGenerator, self).__init__()
        # define a double-convolution block
        def conv_block(in_ch, out_ch, dropout=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout2d(0.3))
            return nn.Sequential(*layers)

        # add AttentionBlock for skip connections
        class AttentionBlock(nn.Module):
            def __init__(self, F_g, F_l, F_int):
                super(AttentionBlock, self).__init__()
                self.W_g = nn.Sequential(
                    nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(F_int)
                )
                self.W_x = nn.Sequential(
                    nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(F_int)
                )
                self.psi = nn.Sequential(
                    nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, g, x):
                g1 = self.W_g(g)
                x1 = self.W_x(x)
                psi = self.relu(g1 + x1)
                psi = self.psi(psi)
                return x * psi

        # Encoder
        self.enc1 = conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_filters*2, base_filters*4, dropout=True)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base_filters*4, base_filters*8, dropout=True)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base_filters*8, base_filters*16, dropout=True)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_filters*16, base_filters*8, dropout=True)
        self.up3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_filters*8, base_filters*4, dropout=True)
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_filters*4, base_filters*2)
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_filters*2, base_filters)

        # Final convolution to produce mask
        self.final = nn.Conv2d(base_filters, 1, kernel_size=1)

        # instantiate attention gates
        self.att4 = AttentionBlock(base_filters*8, base_filters*8, base_filters*4)
        self.att3 = AttentionBlock(base_filters*4, base_filters*4, base_filters*2)
        self.att2 = AttentionBlock(base_filters*2, base_filters*2, base_filters)
        self.att1 = AttentionBlock(base_filters, base_filters, base_filters//2)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)

        u4 = self.up4(b)
        c4 = self.att4(u4, e4)
        d4 = self.dec4(torch.cat([u4, c4], dim=1))

        u3 = self.up3(d4)
        c3 = self.att3(u3, e3)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3)
        c2 = self.att2(u2, e2)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        c1 = self.att1(u1, e1)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        return torch.sigmoid(self.final(d1))

class ResNet18WithMLP(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.resnet = resnet18(num_classes=1000)
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.mlp(x)
        return x
    def get_features(self, x, layer_idx=1):
        x = self.resnet(x)
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == layer_idx * 2 - 1:
                return x
        return x

def initialize_model(config, device):
    """Initialize the interpretable model with pre-trained weights if available"""
    mask_generator = MaskGenerator(in_channels=config['n_channels'])
    classifier = ResNet18WithMLP(num_classes=config['num_classes'], in_channels=config['n_channels'])
    model = InterpretableModel(
        mask_generator=mask_generator,
        classifier=classifier,
        num_classes=config['num_classes'],
        radial_radius=config['radial_radius'],
        radial_decay=config['radial_decay'],
        upper_mask_level_threshold=config['upper_mask_level_threshold'],
        enable_radial_mask_noise=config['enable_radial_mask_noise'],
        radial_mask_noise_range=config['radial_mask_noise_range']
    ).to(device)
    
    # Load pre-trained weights if specified
    if config['pretrained_path']:
        if os.path.exists(config['pretrained_path']):
            try:
                print(f"Loading pretrained classifier weights from {config['pretrained_path']}")
                pretrained_state_dict = torch.load(config['pretrained_path'], map_location=device)
                model.classifier.load_state_dict(pretrained_state_dict)
                
                # Freeze CNN part if specified
                if config['freeze_cnn']:
                    for param in model.classifier.parameters():
                        param.requires_grad = False
                    print("CNN part of the model is frozen")
                else:
                    for param in model.classifier.parameters():
                        param.requires_grad = True
                    print("All classifier parameters unfrozen for finetuning")
                
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {str(e)}")
                print("Using randomly initialized weights instead")
        else:
            print(f"Warning: Pretrained weights not found at {config['pretrained_path']}")
            print("Using randomly initialized weights")
    else:
        print("No pretrained path specified. Using randomly initialized weights")
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MedMNIST interpretable model')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Setup experiment directories
    log_dir, checkpoint_dir, train_viz_dir, val_viz_dir = setup_experiment(config)
    
    # Save config to experiment directory
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA is not available. Training will be extremely slow on CPU.")
        print("Please make sure CUDA is properly installed and configured.")
        device = torch.device("cpu")
    else:
        # Force GPU usage and print device info
        device = torch.device("cuda")
        print("\nGPU Information:")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print(f"\nUsing device: {device}")
    
    # Load MedMNIST dataset
    train_loader, val_loader, test_loader, info = load_medmnist_dataset(config)
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    config['n_channels'] = n_channels
    config['num_classes'] = n_classes
    
    # Get class labels for visualization
    classes = info['label']
    
    # Initialize model
    model = initialize_model(config, device)
    print(f"\nModel device: {next(model.parameters()).device}")
    
    # Initialize loss and optimizer
    criterion = MaskedLoss(
        lambda_mask=config.get('lambda_mask', 1.0),
        lambda_fully_masked=config.get('lambda_fully_masked', 0.2),
        lambda_alignment=config.get('lambda_alignment', 0.5),
        upper_mask_level_threshold=config.get('upper_mask_level_threshold', 0.8),
        dynamic_masked_weight_min=config.get('dynamic_masked_weight_min', 1.0),
        dynamic_masked_weight_max=config.get('dynamic_masked_weight_max', 5.0)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Perform initial evaluation
    #print("\nPerforming initial evaluation...")
    #train_metrics, val_metrics, test_metrics = perform_initial_evaluation(
    #    model, train_loader, val_loader, test_loader, criterion, device, log_dir, classes
    #)
    
    # Train model using the train_model function from toolkits
    print("\nStarting model training...")
    metrics_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        config, log_dir, checkpoint_dir, train_viz_dir, val_viz_dir, classes
    )
    
    # Plot training metrics
    plot_training_metrics(log_dir)
    
    # Final evaluation
    print("\nFINAL EVALUATION")
    print("=" * 50)
    final_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print("\nTraining complete!")
    print(f"Results saved to: {log_dir}")

if __name__ == "__main__":
    main() 