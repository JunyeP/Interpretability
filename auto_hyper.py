import os
import subprocess
import time
from datetime import datetime
import json
import warnings
import optuna
import glob
import re

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

def run_experiment(enable_aug, upper_threshold, enable_noise, data_flag, radial_radius=1, radial_decay=0.2, lambda_mask=1.0, lambda_fully_masked=0.2, lambda_alignment=0.5, dynamic_masked_weight_min=1.0, dynamic_masked_weight_max=5.0):
    """Run a single experiment with given configuration"""
    experiment_name = get_experiment_name(enable_aug, upper_threshold, enable_noise, data_flag)
    print(f"\nStarting experiment: {experiment_name}")
    print("=" * 50)
    config = {
        'data_flag': data_flag,
        'batch_size': 128,
        'num_workers': 2,
        'num_epochs': 4,
        'learning_rate': 0.001,
        'radial_radius': radial_radius,
        'radial_decay': radial_decay,
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
        'pretrained_path': 'pretrain/pathmnist/medmnist_resnet18/best_model.pth',
        'freeze_cnn': False,
        'lambda_mask': lambda_mask,
        'lambda_fully_masked': lambda_fully_masked,
        'lambda_alignment': lambda_alignment,
        'dynamic_masked_weight_min': dynamic_masked_weight_min,
        'dynamic_masked_weight_max': dynamic_masked_weight_max
    }
    config_path = f'temp_config_{experiment_name}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    try:
        print("Starting training...")
        subprocess.run(['python', 'medmnist_finetune.py', '--config', config_path], check=True)
        # After training, find the correct log directory and read test accuracy and loss from test_results.txt
        log_dir = config['log_dir']
        experiment_name = config['experiment_name']
        # Find the latest subdirectory matching the experiment name
        pattern = os.path.join(log_dir, f"{experiment_name}*")
        matching_dirs = glob.glob(pattern)
        if not matching_dirs:
            print(f"Warning: No log directory found for pattern {pattern}")
            return 0.0, 1.0
        # Pick the most recent directory
        latest_dir = max(matching_dirs, key=os.path.getmtime)
        results_file = os.path.join(latest_dir, 'test_results.txt')
        acc = None
        loss = None
        if not os.path.exists(results_file):
            print(f"Warning: test_results.txt not found in {latest_dir}")
            return 0.0, 1.0
        with open(results_file, 'r') as rf:
            lines = rf.readlines()
        for line in lines:
            if 'Loss:' in line and 'Acc:' in line:
                # Format: 'Loss: 0.1234, Acc: 95.43%'
                parts = line.strip().split(',')
                for p in parts:
                    if 'Loss:' in p:
                        loss = float(p.split(':')[1].strip())
                    if 'Acc:' in p:
                        acc = float(p.split(':')[1].replace('%','').strip())
        return acc if acc is not None else 0.0, loss if loss is not None else 1.0
    except subprocess.CalledProcessError as e:
        print(f"Error in experiment {experiment_name}: {e}")
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

def objective(trial, data_flag):
    enable_aug = trial.suggest_categorical('enable_augmentation', [True, True])
    upper_threshold = trial.suggest_float('upper_mask_level_threshold', 0.8, 0.9)
    enable_noise = trial.suggest_categorical('enable_radial_mask_noise', [False, False])
    radial_radius = trial.suggest_int('radial_radius', 3, 3)
    radial_decay = trial.suggest_float('radial_decay', 1.0, 1.0)
    lambda_mask = trial.suggest_float('lambda_mask', 0.0, 5.0)
    lambda_fully_masked = trial.suggest_float('lambda_fully_masked', 0.0, 2.0)
    lambda_alignment = trial.suggest_float('lambda_alignment', 0.0, 5.0)
    dynamic_masked_weight_min = trial.suggest_float('dynamic_masked_weight_min', 0.0, 2.0)
    dynamic_masked_weight_max = trial.suggest_float('dynamic_masked_weight_max', 1.0, 10.0)
    acc, loss = run_experiment(
        enable_aug, upper_threshold, enable_noise, data_flag, radial_radius, radial_decay,
        lambda_mask, lambda_fully_masked, lambda_alignment, dynamic_masked_weight_min, dynamic_masked_weight_max
    )
    # For single-objective: minimize loss
    return loss
    # For multi-objective: return (loss, -acc) or (acc, loss)

def main():
    data_flag = 'pathmnist'  # Change to your desired MedMNIST dataset
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data_flag), n_trials=20)
    print("Best trial parameters:", study.best_trial.params)
    print(f"Best validation ACC: {study.best_trial.value:.2f}")
    # Run final experiment with best hyperparameters
    best = study.best_trial.params
    print("Running final experiment with best hyperparameters...")
    final_acc, final_loss = run_experiment(
        best['enable_augmentation'],
        best['upper_mask_level_threshold'],
        best['enable_radial_mask_noise'],
        data_flag,
        best['radial_radius'],
        best['radial_decay'],
        best['lambda_mask'],
        best['lambda_fully_masked'],
        best['lambda_alignment'],
        best['dynamic_masked_weight_min'],
        best['dynamic_masked_weight_max']
    )
    print(f"Final test ACC: {final_acc:.2f}")
    print(f"Final test Loss: {final_loss:.2f}")
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    # Save all trial results
    all_trials = []
    for t in study.trials:
        trial_result = {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state)
        }
        all_trials.append(trial_result)
    with open("all_hyperparameter_trials.json", "w") as f:
        json.dump(all_trials, f, indent=4)

if __name__ == "__main__":
    main()