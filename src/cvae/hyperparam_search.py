import subprocess
import os
import glob
import json
import csv
import argparse
import random
import numpy as np
from datetime import datetime
import time

GRID_EXPERIMENTS = [
    ("baseline", 128, 0.5, 2e-4, 10),
    ("low_beta", 128, 0.1, 2e-4, 10),
    ("high_beta", 128, 1.0, 2e-4, 10),
    ("high_latent", 256, 0.5, 2e-4, 10),
    ("low_lr", 128, 0.5, 1e-4, 10),
    ("high_lr", 128, 0.5, 5e-4, 10),
    ("long_warmup", 128, 0.5, 2e-4, 15),
    ("short_warmup", 128, 0.5, 2e-4, 5),
    ("combo_low_beta_high_latent", 256, 0.1, 2e-4, 10),
    ("combo_high_latent_stable", 256, 0.5, 1e-4, 10),
]
RANDOM_SEARCH_SPACE = {
    'latent_dim': [64, 128, 256, 512],
    'beta': {
        'type': 'loguniform',
        'low': 0.01,
        'high': 2.0
    },
    'lr': {
        'type': 'loguniform',
        'low': 1e-5,
        'high': 1e-3
    },
    'warmup_epochs': [0, 5, 10, 15, 20]
}

FIXED = {
    'data_root': '',
    'img_size': 64,
    'batch_size': 128,
    'test_split': 0.1,
    'epochs': 30,
    'save_interval': 10,
}


def sample_random_config(search_space, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    config = {}

    for param, space in search_space.items():
        if isinstance(space, list):
            # Categorical: sample from list
            config[param] = random.choice(space)
        elif isinstance(space, dict):
            # Continuous: sample based on type
            if space['type'] == 'uniform':
                config[param] = np.random.uniform(space['low'], space['high'])
            elif space['type'] == 'loguniform':
                config[param] = np.exp(np.random.uniform(
                    np.log(space['low']),
                    np.log(space['high'])
                ))
            elif space['type'] == 'int_uniform':
                config[param] = int(np.random.uniform(space['low'], space['high']))

    return config


def generate_random_experiments(n_trials, seed=None):
    experiments = []

    for i in range(n_trials):
        config = sample_random_config(RANDOM_SEARCH_SPACE, seed=seed + i if seed else None)

        name = f"random_{i + 1:03d}_" + \
               f"lat{config['latent_dim']}_" + \
               f"b{config['beta']:.3f}_" + \
               f"lr{config['lr']:.1e}_" + \
               f"w{config['warmup_epochs']}"

        experiments.append((
            name,
            config['latent_dim'],
            config['beta'],
            config['lr'],
            config['warmup_epochs']
        ))

    return experiments


def run_experiment(exp_name, latent_dim, beta, lr, warmup_epochs):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f"  latent_dim: {latent_dim}")
    print(f"  beta: {beta:.4f}")
    print(f"  lr: {lr:.2e}")
    print(f"  warmup_epochs: {warmup_epochs}")
    print("=" * 70)

    cmd = [
        'python', 'train.py',
        '--output_dir', exp_name,
        '--data_root', FIXED['data_root'],
        '--img_size', str(FIXED['img_size']),
        '--batch_size', str(FIXED['batch_size']),
        '--test_split', str(FIXED['test_split']),
        '--epochs', str(FIXED['epochs']),
        '--save_interval', str(FIXED['save_interval']),
        '--latent_dim', str(latent_dim),
        '--beta', str(beta),
        '--lr', str(lr),
        '--warmup_epochs', str(warmup_epochs),
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, timeout=3600)

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed / 60:.1f} minutes")

        time.sleep(1)  # Wait for filesystem
        possible_dirs = glob.glob(f"*{exp_name}*")

        if not possible_dirs:
            print(f"✗ No output directory found")
            return {
                'name': exp_name,
                'latent_dim': latent_dim,
                'beta': beta,
                'lr': lr,
                'warmup_epochs': warmup_epochs,
                'best_val_loss': float('inf'),
                'status': 'no_dir',
                'elapsed_min': elapsed / 60
            }

        # Look for checkpoint in each directory
        for dir_path in possible_dirs:
            checkpoint_path = os.path.join(dir_path, 'checkpoints', 'best_checkpoint.pt')

            if os.path.exists(checkpoint_path):
                # Load checkpoint
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                best_val_loss = checkpoint.get('min_loss', checkpoint.get('val_loss', float('inf')))

                print(f"✓ Found checkpoint! Best val loss: {best_val_loss:.4f}")

                return {
                    'name': exp_name,
                    'latent_dim': latent_dim,
                    'beta': float(beta),
                    'lr': float(lr),
                    'warmup_epochs': warmup_epochs,
                    'best_val_loss': float(best_val_loss),
                    'final_train_loss': float(checkpoint.get('train_loss', 0)),
                    'final_val_loss': float(checkpoint.get('val_loss', 0)),
                    'status': 'completed',
                    'checkpoint': checkpoint_path,
                    'elapsed_min': elapsed / 60
                }

        print(f"✗ No checkpoint found in directories: {possible_dirs}")
        return {
            'name': exp_name,
            'latent_dim': latent_dim,
            'beta': float(beta),
            'lr': float(lr),
            'warmup_epochs': warmup_epochs,
            'best_val_loss': float('inf'),
            'status': 'no_checkpoint',
            'elapsed_min': elapsed / 60
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ Timeout after {elapsed / 60:.1f} minutes")
        return {
            'name': exp_name,
            'latent_dim': latent_dim,
            'beta': float(beta),
            'lr': float(lr),
            'warmup_epochs': warmup_epochs,
            'best_val_loss': float('inf'),
            'status': 'timeout',
            'elapsed_min': elapsed / 60
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error: {e}")
        return {
            'name': exp_name,
            'latent_dim': latent_dim,
            'beta': float(beta),
            'lr': float(lr),
            'warmup_epochs': warmup_epochs,
            'best_val_loss': float('inf'),
            'status': f'error: {str(e)}',
            'elapsed_min': elapsed / 60
        }


def save_results(results_dir, results):
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)


def print_summary(results, results_dir):
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)

    completed = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] != 'completed']

    print(f"\nTotal experiments: {len(results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['name']}: {r['status']}")

    if completed:
        completed.sort(key=lambda x: x['best_val_loss'])

        print("\n" + "-" * 70)
        print("Top 5 Results:")
        print("-" * 70)
        for i, r in enumerate(completed[:5], 1):
            print(f"{i}. {r['name']}")
            print(f"   Val Loss: {r['best_val_loss']:.4f}")
            print(f"   Config: latent={r['latent_dim']}, beta={r['beta']:.4f}, "
                  f"lr={r['lr']:.2e}, warmup={r['warmup_epochs']}")
            print(f"   Time: {r['elapsed_min']:.1f} min")
            print()

        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        best = completed[0]
        print(f"  Name: {best['name']}")
        print(f"  Validation Loss: {best['best_val_loss']:.4f}")
        print(f"  Latent Dim: {best['latent_dim']}")
        print(f"  Beta: {best['beta']:.4f}")
        print(f"  Learning Rate: {best['lr']:.2e}")
        print(f"  Warmup Epochs: {best['warmup_epochs']}")
        print(f"  Training Time: {best['elapsed_min']:.1f} minutes")

        print(f"\nTo reproduce this configuration:")
        print(f"python train.py \\")
        print(f"    --latent_dim {best['latent_dim']} \\")
        print(f"    --beta {best['beta']:.4f} \\")
        print(f"    --lr {best['lr']:.2e} \\")
        print(f"    --warmup_epochs {best['warmup_epochs']} \\")
        print(f"    --epochs 100")
    else:
        print("\n✗ No experiments completed successfully")
        print("Check the logs for errors")

    print(f"\nResults saved to: {results_dir}/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for CLIP-CVAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--mode", type=str, default="grid",
                        choices=['grid', 'random'],
                        help="Search mode: grid or random")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random trials (only for random mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--data_root", type=str, default="data/processed",
                        help="Root directory containing training data")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout in seconds per experiment")

    args = parser.parse_args()

    # Update fixed parameters
    FIXED['data_root'] = args.data_root
    FIXED['epochs'] = args.epochs
    FIXED['batch_size'] = args.batch_size

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Generate experiments based on mode
    if args.mode == 'grid':
        experiments = GRID_EXPERIMENTS
        mode_str = "GRID"
    else:
        experiments = generate_random_experiments(args.n_trials, seed=args.seed)
        mode_str = "RANDOM"

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hypersearch_{args.mode}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save search configuration
    config = {
        'mode': args.mode,
        'n_trials': len(experiments),
        'seed': args.seed,
        'fixed_params': FIXED,
        'timestamp': timestamp
    }
    if args.mode == 'random':
        config['search_space'] = RANDOM_SEARCH_SPACE

    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print(f"{mode_str} HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Epochs per experiment: {FIXED['epochs']}")
    print(f"Batch size: {FIXED['batch_size']}")
    if args.mode == 'random':
        print(f"Random trials: {args.n_trials}")
        if args.seed:
            print(f"Random seed: {args.seed}")
    print("=" * 70)

    results = []

    # Run all experiments
    for i, (name, latent_dim, beta, lr, warmup) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment: {name}")

        result = run_experiment(name, latent_dim, beta, lr, warmup)
        results.append(result)

        # Save intermediate results
        save_results(results_dir, results)

        # Print current standings
        completed = [r for r in results if r['status'] == 'completed']
        if completed:
            best = min(completed, key=lambda x: x['best_val_loss'])
            print(f"\n✓ Best so far: {best['name']} with val_loss={best['best_val_loss']:.4f}")

    # Final summary
    print_summary(results, results_dir)


if __name__ == "__main__":
    main()