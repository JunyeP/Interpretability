import os
import json
import subprocess
from pathlib import Path
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

def find_experiment_folders(logs_dir):
    """Find all experiment folders in the logs directory."""
    experiment_folders = []
    for item in os.listdir(logs_dir):
        item_path = os.path.join(logs_dir, item)
        if os.path.isdir(item_path) and item.startswith('cifar10_finetune'):
            # Find the timestamped subfolder
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and subitem.startswith(item):
                    experiment_folders.append(subitem_path)
    return experiment_folders

def get_experiment_config(experiment_path):
    """Read the config.json file from an experiment folder."""
    config_path = os.path.join(experiment_path, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def run_evaluation(experiment_path, config):
    """Run evaluate_cifar10.py for a single experiment."""
    # Construct the paths
    pretrained_path = './pretrained_best_model.pth'
    interpretable_path = os.path.join(experiment_path, 'checkpoints', 'final_epoch.pth')
    
    # Extract hyperparameters from config and convert radius to integer
    radial_radius = int(config['radial_radius'])  # Convert to integer
    radial_decay = config['radial_decay']
    upper_mask_level_threshold = config['upper_mask_level_threshold']
    
    # Run the evaluation script
    cmd = [
        'python', 'evaluate_cifar10.py',
        '--experiment_path', experiment_path,
        '--pretrained_path', pretrained_path,
        '--interpretable_path', interpretable_path,
        '--radial_radius', str(radial_radius),
        '--radial_decay', str(radial_decay),
        '--upper_mask_level_threshold', str(upper_mask_level_threshold)
    ]
    
    print(f"\nRunning evaluation for experiment: {os.path.basename(experiment_path)}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Evaluation completed successfully for {experiment_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {experiment_path}: {e}")

def main():
    logs_dir = './logs'
    experiment_folders = find_experiment_folders(logs_dir)
    
    print(f"Found {len(experiment_folders)} experiment folders to evaluate")
    
    for experiment_path in experiment_folders:
        try:
            config = get_experiment_config(experiment_path)
            run_evaluation(experiment_path, config)
        except Exception as e:
            print(f"Error processing experiment {experiment_path}: {e}")

if __name__ == "__main__":
    main()
