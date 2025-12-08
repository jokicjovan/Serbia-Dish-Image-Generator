"""
Optimized hyperparameters for training GAN on T4 GPU with 7000 images.

T4 GPU Specs:
- Memory: 16GB GDDR6
- CUDA Cores: 2560
- Memory Bandwidth: 320 GB/s
- Recommended for: Medium-scale training with memory efficiency

Dataset: 7000 images with captions
"""

# T4 Optimized Training Configuration
T4_CONFIG = {
    # ===== Dataset & Memory Settings =====
    'batch': 16,              # Reduced from 32 for T4 memory constraints
    'num_workers': 4,         # Good for T4's CPU interface
    'pin_memory': True,       # Faster CPU->GPU transfer
    'img_size': 128,          # Keep at 128x128 for balance of quality/memory

    # ===== Model Architecture =====
    'z_dim': 128,             # Standard latent dimension
    'cond_dim': 256,          # Conditioning projection dimension
    'base_ch': 64,            # Base channels (memory efficient)

    # ===== Training Duration (for 7000 images) =====
    'iters': 50000,           # Reduced from 200k - smaller dataset converges faster
    'epochs_approx': 114,     # Approximate epochs (50000 / (7000/16))

    # ===== Learning Rates & Optimization =====
    'g_lr': 1e-4,             # Slightly lower G lr for stability with smaller dataset
    'd_lr': 2e-4,             # Slightly higher D lr to prevent mode collapse
    'betas': (0.0, 0.9),      # Standard for GANs
    'weight_decay': 1e-5,     # Light regularization

    # ===== Training Balance =====
    'n_disc': 2,              # More D updates to stabilize training
    'r1_gamma': 5.0,          # Stronger R1 penalty for smaller dataset
    'r1_every': 8,            # More frequent R1 (was 16) for stability

    # ===== Regularization & Stability =====
    'ema': 0.9995,            # Slightly faster EMA decay for smaller dataset
    'diffaugment': True,      # CRITICAL for small datasets - prevents overfitting
    'use_mismatch': True,     # Helps discriminator learn better
    'grad_clip_norm': 10.0,   # Gradient clipping for stability

    # ===== Monitoring & Checkpointing =====
    'sample_every': 1000,     # Sample every 1k steps
    'ckpt_every': 5000,       # Checkpoint every 5k steps
    'n_sample': 16,           # Sample grid size

    # ===== T4-Specific Optimizations =====
    'mixed_precision': True,  # Use AMP for memory efficiency
    'compile_model': True,    # Use torch.compile for T4 optimization
    'channels_last': True,    # Memory layout optimization
    'benchmark_cudnn': True,  # Optimize CUDNN for consistent input sizes
}

# Advanced T4 Configuration with Adaptive Training
T4_ADAPTIVE_CONFIG = {
    **T4_CONFIG,  # Inherit base config

    # ===== Adaptive Training Controls =====
    'adaptive_training': True,
    'target_hinge': 1.4,      # Lower target for smaller dataset
    'hi_margin': 0.3,         # Tighter margins for more responsive control
    'lo_margin': 0.7,
    'd_lr_step': 1.1,         # Gentler learning rate adjustments
    'd_lr_min': 5e-5,
    'd_lr_max': 5e-4,
    'n_disc_min': 1,
    'n_disc_max': 3,
    'window': 150,            # Shorter EWMA window for responsiveness
    'cool_down': 300,         # Shorter cooldown for smaller dataset
}

# Memory-Optimized T4 Configuration (if hitting memory limits)
T4_MEMORY_EFFICIENT = {
    **T4_CONFIG,
    'batch': 12,              # Further reduced batch size
    'base_ch': 48,            # Smaller base channels
    'mixed_precision': True,  # Essential for memory savings
    'gradient_checkpointing': True,  # Trade compute for memory
    'efficient_attention': True,     # If using attention layers
}

def get_t4_config(dataset_size=7000, memory_mode='standard'):
    """
    Get optimized configuration for T4 training.

    Args:
        dataset_size (int): Number of training images
        memory_mode (str): 'standard', 'adaptive', or 'efficient'

    Returns:
        dict: Training configuration
    """
    if memory_mode == 'adaptive':
        config = T4_ADAPTIVE_CONFIG.copy()
    elif memory_mode == 'efficient':
        config = T4_MEMORY_EFFICIENT.copy()
    else:
        config = T4_CONFIG.copy()

    # Adjust iterations based on dataset size
    if dataset_size < 5000:
        config['iters'] = 30000
        config['r1_gamma'] = 8.0  # Stronger regularization for very small datasets
    elif dataset_size > 10000:
        config['iters'] = 75000
        config['r1_gamma'] = 3.0  # Lighter regularization for larger datasets

    return config

# Usage examples in comments
"""
# Standard T4 training
config = get_t4_config(dataset_size=7000, memory_mode='standard')

# With adaptive training controller
config = get_t4_config(dataset_size=7000, memory_mode='adaptive')

# Memory-constrained training
config = get_t4_config(dataset_size=7000, memory_mode='efficient')

# Command line usage:
python train.py --batch 16 --iters 50000 --g_lr 1e-4 --d_lr 2e-4 \
    --n_disc 2 --r1_gamma 5.0 --r1_every 8 --diffaugment --use_mismatch \
    --ema 0.9995 --sample_every 1000 --ckpt_every 5000

# Or with dynamic training:
python dynamic_train.py --batch 16 --iters 50000 --g_lr 1e-4 --d_lr 2e-4 \
    --adaptive --target_hinge 1.4 --window 150 --cool_down 300
"""