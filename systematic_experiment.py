import os
import subprocess
import time
from datetime import datetime
import json
import warnings

# Suppress specific PyTorch deprecation warning
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

def get_experiment_name(enable_aug, upper_threshold, enable_noise, data_flag):
    """Generate experiment name based on configuration"""
    name_parts = [f'{data_flag}_finetune']
    if enable_aug:
        name_parts.append('aug')
    else:
        name_parts.append('noaug')
    name_parts.append(f'thresh{upper_threshold}')
    if enable_noise:
        name_parts.append('noise')
    else:
        name_parts.append('nonoise')
    return '_'.join(name_parts)

def run_experiment(enable_aug, upper_threshold, enable_noise, data_flag):
    """Run a single experiment with given configuration"""
    experiment_name = get_experiment_name(enable_aug, upper_threshold, enable_noise, data_flag)
    print(f"\nStarting experiment: {experiment_name}")
    print("=" * 50)
    config = {
        'data_flag': data_flag,
        'batch_size': 64,
        'num_workers': 2,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'radial_radius': 1,
        'radial_decay': 0.2,
        'upper_mask_level_threshold': upper_threshold,
        'enable_augmentation': enable_aug,
        'augmentation_transforms': {
            'random_crop': True,
            'random_horizontal_flip': True,
            'random_rotation': 15,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            }
        },
        'enable_radial_mask_noise': enable_noise,
        'radial_mask_noise_range': 0.2,
        'experiment_name': experiment_name,
        'log_dir': './logs',
        'pretrained_path': './pretrained_best_model.pth',
        'freeze_cnn': False
    }
    config_path = f'temp_config_{experiment_name}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    try:
        print("Starting training...")
        subprocess.run(['python', 'medmnist_finetune.py', '--config', config_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in experiment {experiment_name}: {e}")
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

def main():
    data_flag = 'pathmnist'  # Change to your desired MedMNIST dataset
    combinations = [
        (True, 0.6, True),   # With aug, thresh 0.6, with noise
        (True, 0.6, False),  # With aug, thresh 0.6, no noise
        (True, 0.8, True),   # With aug, thresh 0.8, with noise
        (True, 0.8, False),  # With aug, thresh 0.8, no noise
        (False, 0.6, True),  # No aug, thresh 0.6, with noise
        (False, 0.6, False), # No aug, thresh 0.6, no noise
        (False, 0.8, True),  # No aug, thresh 0.8, with noise
        (False, 0.8, False)  # No aug, thresh 0.8, no noise
    ]
    for enable_aug, upper_threshold, enable_noise in combinations:
        start_time = time.time()
        run_experiment(enable_aug, upper_threshold, enable_noise, data_flag)
        end_time = time.time()
        print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")
        print("=" * 50)

if __name__ == "__main__":
    main() 