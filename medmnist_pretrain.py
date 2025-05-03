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
import sys
import warnings
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
from torchvision.models import resnet18
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Suppress specific PyTorch deprecation warning
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

def load_config():
    """Load configuration parameters for MedMNIST"""
    return {
        'data_flag': 'pathmnist',  # Change to your desired MedMNIST dataset
        'batch_size': 64,
        'num_workers': 2,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'experiment_name': 'medmnist_resnet18',
    }

def load_medmnist_dataset(config):
    data_flag = config['data_flag']
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']
    img_size = 224

    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5]*n_channels, std=[.5]*n_channels)
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader, test_loader, info

def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return (predicted == labels).sum().item() / labels.size(0) * 100

def validate_model(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_acc += get_accuracy(outputs, labels) * batch_size
            total_samples += batch_size
    val_loss = val_loss / total_samples
    val_acc = val_acc / total_samples
    return val_loss, val_acc

def plot_metrics(train_losses, train_accs, val_losses, val_accs, log_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
    plt.close()

class ResNet18WithMLP(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.resnet = resnet18(num_classes=1000)  # We'll ignore the original FC
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the original FC layer
        self.resnet.fc = nn.Identity()
        # Custom MLP head
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
        # Extract features from specified MLP layer
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == layer_idx * 2 - 1:  # ReLU after linear layer
                return x
        return x

def main():
    config = load_config()
    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config['experiment_name']
    log_dir = os.path.join("pretrain", config['data_flag'], experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {config['data_flag']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Number of Epochs: {config['num_epochs']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Device: {device}")
    train_loader, val_loader, test_loader, info = load_medmnist_dataset(config)
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    img_size = 224
    model = ResNet18WithMLP(num_classes=n_classes, in_channels=n_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    print(f"\nStarting MedMNIST Training")
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_samples = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = get_accuracy(outputs, labels)
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_acc += acc * batch_size
            total_samples += batch_size
            train_loader_tqdm.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{acc:.2f}%"
            )
        train_loss = train_loss / total_samples
        train_acc = train_acc / total_samples
        val_loss, val_acc = validate_model(model, criterion, val_loader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"  TRAIN - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  VAL   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print(f"  SAVED BEST MODEL with validation accuracy: {best_val_acc:.2f}%")
    # Final test evaluation and MedMNIST metrics
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            outputs = outputs.softmax(dim=-1)
            y_score = torch.cat((y_score, outputs.cpu()), 0)
            y_true = torch.cat((y_true, labels.cpu()), 0)
    y_score = y_score.detach().numpy()
    y_true = y_true.detach().numpy()
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    # Compute test metrics manually using scikit-learn
    auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    acc = accuracy_score(y_true, y_score.argmax(axis=1))
    print(f'test  auc: {auc:.3f}  acc: {acc:.3f}')
    test_loss, test_acc = validate_model(model, criterion, test_loader, device)
    print(f"\nTEST RESULTS:")
    print(f"  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    with open(os.path.join(log_dir, "test_results.txt"), "w") as f:
        f.write(f"TEST RESULTS:\n")
        f.write(f"Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n")
        f.write(f"AUC: {auc:.3f}, ACC: {acc:.3f}\n")
    plot_metrics(train_losses, train_accs, val_losses, val_accs, log_dir)
    print(f"\nTraining complete!")
    # Save best and final metrics to a summary file
    summary_path = os.path.join(log_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {val_accs[-1]:.4f}\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Final Test AUC: {auc:.4f}\n")
    print(f"\nSummary written to {summary_path}")

if __name__ == "__main__":
    print(INFO['pathmnist'])
    main()
