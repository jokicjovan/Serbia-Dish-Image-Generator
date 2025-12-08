# Serbian Dish GAN - Improved Training Guide

## 🚀 Recent Improvements (Fixed Critical Issues)

Your GAN implementation has been **significantly improved** with the following critical fixes:

### ✅ Critical Bugs Fixed
1. **Embedding Dimension Mismatch** - Fixed hardcoded 512-dim in inference scripts
2. **Broken Constructors** - Fixed `_init_` → `__init__` in dynamic_train.py
3. **Missing EMA Method** - Added `copy_to()` method for dynamic training compatibility
4. **Poor Mismatch Generation** - Replaced fixed roll with random permutation

### 🎯 T4 GPU Optimizations
5. **T4-Specific Configuration** - Optimized hyperparameters for 16GB memory + 7000 images
6. **Gradient Clipping** - Added stability features for small dataset training
7. **Better Error Handling** - Improved dataset loading with descriptive error messages
8. **Code Cleanup** - Renamed duplicate scripts and organized codebase

---

## 🏃‍♂️ Quick Start for T4 Training

### 1. Recommended T4 Training Command

For your 7000 image dataset:

```bash
# Standard T4 training (recommended)
python train.py \
    --batch 16 \
    --iters 50000 \
    --g_lr 1e-4 \
    --d_lr 2e-4 \
    --n_disc 2 \
    --r1_gamma 5.0 \
    --r1_every 8 \
    --grad_clip 10.0 \
    --diffaugment \
    --use_mismatch \
    --ema 0.9995 \
    --sample_every 1000 \
    --ckpt_every 5000 \
    --data_root data/processed \
    --out_dir runs/t4_training
```

### 2. Advanced T4 Training (with adaptive control)

```bash
# Adaptive training with dynamic learning rate control
python dynamic_train.py \
    --batch 16 \
    --iters 50000 \
    --g_lr 1e-4 \
    --d_lr 2e-4 \
    --grad_clip 10.0 \
    --adaptive \
    --target_hinge 1.4 \
    --window 150 \
    --cool_down 300 \
    --data_root data/processed \
    --out_dir runs/t4_adaptive
```

### 3. Memory-Efficient T4 Training (if hitting memory limits)

```bash
# Reduced batch size and model size for memory constraints
python train.py \
    --batch 12 \
    --base_ch 48 \
    --iters 50000 \
    --grad_clip 10.0 \
    --diffaugment \
    --use_mismatch \
    --data_root data/processed \
    --out_dir runs/t4_efficient
```

---

## 🎛️ Key Training Parameters Explained

| Parameter | T4 Optimized | Purpose |
|-----------|--------------|---------|
| `--batch` | 16 (was 32) | Fits T4 16GB memory |
| `--iters` | 50,000 (was 200k) | Faster convergence for 7k dataset |
| `--g_lr` | 1e-4 (was 2e-4) | Lower G lr for stability |
| `--d_lr` | 2e-4 | Slightly higher D lr prevents collapse |
| `--n_disc` | 2 (was 1) | More D updates for balance |
| `--r1_gamma` | 5.0 (was 1.0) | Stronger regularization for small dataset |
| `--r1_every` | 8 (was 16) | More frequent penalty application |
| `--grad_clip` | 10.0 | **NEW**: Gradient clipping for stability |
| `--diffaugment` | True | **CRITICAL** for small datasets |
| `--use_mismatch` | True | Helps D learn better text alignment |

---

## 🧪 Training Monitoring

### Expected Timeline (T4 with 7000 images)
- **Total Training**: ~50,000 iterations ≈ **6-8 hours**
- **First samples**: Visible improvement by 5,000 steps
- **Good quality**: 25,000+ steps
- **Best results**: 40,000+ steps

### Monitoring Progress
```bash
# Watch training logs
tail -f runs/t4_training/log.txt

# Check sample grids (generated every 1000 steps)
ls runs/t4_training/samples/

# Monitor checkpoints (saved every 5000 steps)
ls runs/t4_training/ckpt_*.pt
```

### Training Curves to Watch
- **D Loss**: Should stabilize around 1.0-2.0
- **G Loss**: Should decrease and stabilize around -1 to -5
- **R1 Penalty**: Applied every 8 steps, should be moderate

---

## 🎨 Inference & Generation

### 1. Generate from Embeddings
```bash
# Generate from specific embedding file
python inference.py \
    --checkpoint runs/t4_training/ckpt_050000.pt \
    --embedding data/processed/embedds/dish_001.npy \
    --num_samples 16 \
    --output_dir generated/dish_001

# Generate from all embeddings in directory
python inference.py \
    --checkpoint runs/t4_training/ckpt_050000.pt \
    --embedding_dir data/processed/embedds \
    --num_samples 8 \
    --output_dir generated/all_dishes
```

### 2. Generate from Text Prompts (Requires CLIP)
```bash
# Install CLIP first
pip install git+https://github.com/openai/CLIP.git

# Single prompt
python inference_prompt.py \
    --checkpoint runs/t4_training/ckpt_050000.pt \
    --prompt "a delicious Serbian ćevapi with onions" \
    --num_samples 16 \
    --output_dir generated/prompts

# Interactive mode
python inference_prompt.py \
    --checkpoint runs/t4_training/ckpt_050000.pt \
    --interactive
```

### 3. Batch Generation from File
```bash
# Create prompts.txt with one prompt per line:
echo "Serbian burek with cheese" > prompts.txt
echo "grilled meat platter with vegetables" >> prompts.txt
echo "traditional Serbian dessert" >> prompts.txt

python inference_prompt.py \
    --checkpoint runs/t4_training/ckpt_050000.pt \
    --prompts_file prompts.txt \
    --num_samples 8 \
    --output_dir generated/batch
```

---

## 🔧 Configuration Files

### Using T4 Optimized Config
```python
from t4_config import get_t4_config

# Standard T4 config
config = get_t4_config(dataset_size=7000, memory_mode='standard')

# Adaptive training config
config = get_t4_config(dataset_size=7000, memory_mode='adaptive')

# Memory-efficient config
config = get_t4_config(dataset_size=7000, memory_mode='efficient')
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Memory Issues
```bash
# Reduce batch size
--batch 12

# Reduce model size
--base_ch 48

# Enable memory optimizations (if available)
--mixed_precision
```

#### Training Instability
```bash
# Increase gradient clipping
--grad_clip 15.0

# Stronger regularization
--r1_gamma 8.0

# More frequent R1 penalty
--r1_every 4
```

#### Slow Convergence
```bash
# More discriminator updates
--n_disc 3

# Differential augmentation (essential!)
--diffaugment

# Use mismatch loss
--use_mismatch
```

#### Dimension Mismatch Errors
- **Fixed!** Inference scripts now automatically detect embedding dimensions
- Check that your `.npy` files contain valid CLIP embeddings
- Standard CLIP ViT-B/32 = 512 dims, ViT-L/14 = 768 dims

---

## 📊 Expected Results

### Quality Progression
- **5k steps**: Basic shapes and colors
- **15k steps**: Recognizable food items
- **30k steps**: Good texture and details
- **50k steps**: High-quality, diverse samples

### Success Indicators
- ✅ Generator loss decreasing and stable
- ✅ Discriminator loss around 1.0-2.0
- ✅ Sample diversity improving over time
- ✅ Generated images match text descriptions
- ✅ No mode collapse (variety in samples)

---

## 🚀 Next Steps

1. **Monitor first 5k steps** - Check if training is stable
2. **Evaluate at 25k steps** - Assess quality and decide if adjustments needed
3. **Full training to 50k** - Complete training for best results
4. **Experiment with prompts** - Test text-to-image generation capabilities

**Happy Training! 🎉**