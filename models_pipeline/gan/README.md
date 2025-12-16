# Serbian Dish Image Generator - GAN Implementation

This directory contains a conditional GAN system for generating Serbian food images from text prompts using CLIP embeddings.

## 📚 Theory & Background

### Generative Adversarial Networks (GANs)

GANs consist of two neural networks competing in a minimax game:

**Generator (G)**: Learns to create realistic images from random noise
**Discriminator (D)**: Learns to distinguish real images from generated ones

The training objective:
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Where:
- `x` = real images
- `z` = random noise vector
- `G(z)` = generated images
- `D(x)` = discriminator's probability that x is real

### Conditional GANs (cGANs)

Standard GANs generate random samples. Conditional GANs add control by conditioning both networks on additional information (labels, text, etc.):

```
min_G max_D V(D,G) = E_x,c[log D(x,c)] + E_z,c[log(1 - D(G(z,c),c))]
```

Where `c` = conditioning information (in our case, CLIP text embeddings)

### CLIP (Contrastive Language-Image Pre-training)

CLIP learns joint representations of text and images by training on 400M+ image-text pairs:

**Key Concepts:**
- **Contrastive Learning**: Learns to match corresponding image-text pairs while separating mismatched pairs
- **Zero-shot Transfer**: Can understand new concepts without specific training
- **Embedding Space**: Maps both text and images to a shared 512D or 768D vector space

**Architecture Options:**
- **ViT-B/32**: 512-dimensional embeddings, faster but lower quality
- **ViT-L/14**: 768-dimensional embeddings, slower but higher quality text understanding

### Text-to-Image Synthesis Challenges

**1. Semantic Understanding**
- Model must understand food terminology ("ćevapi", "burek", "ajvar")
- Spatial relationships ("pizza with olives on top")
- Cultural context (Serbian presentation styles)

**2. Visual Consistency**
- Generated images must match text descriptions
- Avoid mode collapse (generating identical images)
- Maintain realistic textures and colors

**3. Training Stability**
- GANs are notoriously unstable to train
- Discriminator can become too strong, making generator unable to improve
- Requires careful hyperparameter tuning

### Advanced Training Techniques

**1. Spectral Normalization**
- Constrains Lipschitz constant of discriminator
- Improves training stability by preventing discriminator from becoming too strong
- Applied to all discriminator convolutional layers

**2. R1 Gradient Penalty**
- Regularizes discriminator gradients around real data manifold
- Formula: `λ/2 * ||∇D(x)||²` where λ is penalty weight
- Prevents discriminator gradients from exploding

**3. Exponential Moving Average (EMA)**
- Maintains running average of generator weights: `θ_ema = β*θ_ema + (1-β)*θ_current`
- Used for inference to get more stable, higher-quality outputs
- Reduces noise in generated samples
- **Implementation Details:**
  - **Decay Rate**: 0.999 provides very conservative averaging (99.9% old weights, 0.1% new)
  - **Update Frequency**: Updated every training step after generator optimization
  - **Storage**: EMA weights stored separately from training weights in `ema.shadow`
  - **Inference Usage**: EMA weights applied to generator during inference for cleaner outputs
  - **Backup/Restore**: Training weights backed up when EMA applied, restored after inference
- **Benefits:**
  - **Stability**: Smooths out training fluctuations and gradient noise
  - **Quality**: Produces visually cleaner, more coherent images
  - **Robustness**: Less sensitive to individual bad training steps
  - **Convergence**: Helps generator converge to better local optima
- **Alternative Decay Rates:**
  - 0.9995: More stable, slower adaptation to changes
  - 0.995: Faster adaptation, less smoothing
  - Adaptive: `decay = min(0.999, (1 + step) / (10 + step))` for gradual increase

**4. Differentiable Augmentation**
- Applies random augmentations to both real and fake images during training
- Prevents discriminator from overfitting to training data artifacts
- Includes color jittering, random crops, and flips

**5. Hinge Loss**
- Alternative to standard GAN loss, often more stable:
- `L_D = -E[min(0, -1 + D(x))] - E[min(0, -1 - D(G(z)))]`
- `L_G = -E[D(G(z))]`

### Conditional Batch Normalization

Standard Batch Normalization:
```
y = γ * (x - μ) / σ + β
```

Conditional Batch Normalization uses text embeddings to modulate normalization:
```
γ = MLP_γ(text_embedding)
β = MLP_β(text_embedding)
```

This allows the generator to adapt its feature processing based on text input, enabling text-controlled generation.

### Architecture Design Principles

**Generator Design:**
- **Noise Input**: Random 128D vector provides stochasticity
- **Text Conditioning**: CLIP embeddings guide content generation
- **Progressive Upsampling**: 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
- **Residual Connections**: Help gradient flow in deep networks
- **Conditional BatchNorm**: Text-adaptive feature normalization

**Discriminator Design:**
- **Multi-scale Processing**: Operates on multiple image resolutions
- **Conditional Input**: Receives both image and text embedding
- **Spectral Normalization**: Maintains training stability
- **Progressive Downsampling**: 128×128 → 64×64 → ... → 4×4 → scalar

### Loss Functions & Training Dynamics

**Generator Loss:**
- Adversarial loss: Fool discriminator
- Optional: Feature matching loss, perceptual loss

**Discriminator Loss:**
- Real image classification
- Fake image detection
- R1 gradient penalty for regularization

**Training Balance:**
- Generator updates: 1 per iteration
- Discriminator updates: 1-2 per iteration (depending on relative strength)
- Learning rates: Usually D_lr > G_lr for stability

### Embedding Quality Importance

**Why 768D > 512D:**
- **Semantic Richness**: More dimensions capture finer semantic distinctions
- **Food-specific Terms**: Better understanding of culinary vocabulary
- **Cultural Context**: Improved comprehension of regional food terms
- **Spatial Relationships**: Better encoding of object arrangements

**CLIP Model Comparison:**
```
ViT-B/32 (512D): Fast, basic understanding
ViT-L/14 (768D): Slower, nuanced understanding, better for specialized domains
ViT-H/14 (1024D): Slowest, highest quality (if you have compute budget)
```

### Quality Metrics & Evaluation

**Quantitative Metrics:**
- **FID (Fréchet Inception Distance)**: Measures distribution similarity between real and generated images
- **IS (Inception Score)**: Evaluates both quality and diversity
- **CLIP Score**: Measures text-image alignment using CLIP embeddings

**Qualitative Evaluation:**
- Visual inspection of sample quality
- Text-image semantic alignment
- Diversity of generated samples
- Absence of mode collapse artifacts

## 📁 File Structure

### Core Files (Essential)

- **`models.py`** - GAN architecture (Generator, Discriminator)
- **`dataset.py`** - Data loading for images and CLIP embeddings
- **`train_with_plotting.py`** - Main training script with loss visualization
- **`inference_prompt.py`** - Text-to-image generation using CLIP
- **`ema.py`** - Exponential Moving Average for stable training
- **`diffaug.py`** - Differentiable data augmentation

### Optional/Diagnostic Files

- **`analyze_embeddings.py`** - CLIP model compatibility checker (can delete)
- **`inference.py`** - Direct embedding inference (can delete if using text prompts)
- **`train.py`** - Basic training without plotting (redundant)
- **`t4_config.py`** - T4 GPU configs (can delete)
- **`dynamic_train.py`** - Experimental adaptive training (can delete)

## 🏗️ Architecture Overview

### Generator
- Input: Random noise (z_dim=128) + CLIP text embedding (768D for ViT-L/14)
- Architecture: ResNet-style with conditional batch normalization
- Output: 128x128 RGB images normalized to [-1, 1]

### Discriminator
- Input: Images + CLIP text embeddings for conditioning
- Architecture: Convolutional with spectral normalization
- Output: Real/fake logits for adversarial training

### CLIP Integration
- Model: **ViT-L/14** with **laion2b_s32b_b82k** pretrained weights
- Embedding dim: **768D** (not the default 512D)
- Used for text conditioning during training and inference

## 🚀 How We Ran The Training

### 1. Data Preparation
```bash
# Enhanced CLIP embeddings were generated using:
python src/processing/make_enhanced_clip_embeds.py \
    --input_dir data/processed/captions \
    --output_dir data/processed/enhanced_768d_embeds \
    --model ViT-L-14 \
    --pretrained laion2b_s32b_b82k
```

### 2. Training Command (Final Stable Version)
```bash
python src/gan/train_with_plotting.py \
    --root data/processed \
    --embeddings_dir enhanced_768d_embeds \
    --out_dir runs/cgan_768d \
    --batch 4 \
    --base_ch 32 \
    --cond_dim 768 \
    --lr_g 0.00005 \
    --lr_d 0.00015 \
    --r1_penalty 2 \
    --grad_clip 1.0 \
    --iters 100000
```

### 3. Resuming from Checkpoint
```bash
python src/gan/train_with_plotting.py \
    --root data/processed \
    --embeddings_dir enhanced_768d_embeds \
    --out_dir runs/cgan_768d_continued \
    --batch 4 \
    --base_ch 32 \
    --cond_dim 768 \
    --lr_g 0.00005 \
    --lr_d 0.00015 \
    --r1_penalty 2 \
    --grad_clip 1.0 \
    --resume_from runs/cgan_768d/ckpt_0020000.pt \
    --iters 100000
```

## 📊 Training Parameters Explanation

### Key Hyperparameters
- **`--batch 4`** - Small batch size for T4 GPU memory constraints
- **`--base_ch 32`** - Reduced channel count to fit in GPU memory
- **`--cond_dim 768`** - CLIP ViT-L/14 embedding dimension
- **`--lr_g 0.00005`** - Conservative generator learning rate
- **`--lr_d 0.00015`** - Conservative discriminator learning rate
- **`--r1_penalty 2`** - R1 gradient penalty for discriminator regularization
- **`--grad_clip 1.0`** - Gradient clipping to prevent instability

### Why These Values?
- **Small batch/channels**: T4 GPU has limited 16GB memory
- **768D embeddings**: ViT-L/14 produces higher quality text representations than ViT-B/32 (512D)
- **Conservative LRs**: Prevents training instability and discriminator overpowering
- **Gradient clipping**: Essential for stable GAN training

## 🎨 Inference (Generating Images)

### Text-to-Image Generation
```bash
# Single prompt
python src/gan/inference_prompt.py \
    --checkpoint runs/cgan_768d/ckpt_0050000.pt \
    --prompt "traditional Serbian ćevapi with onions"

# Interactive mode
python src/gan/inference_prompt.py \
    --checkpoint runs/cgan_768d/ckpt_0050000.pt \
    --interactive

# Multiple prompts from file
python src/gan/inference_prompt.py \
    --checkpoint runs/cgan_768d/ckpt_0050000.pt \
    --prompts_file prompts.txt
```

### Key Features
- **Automatic CLIP model detection** from checkpoint embedding dimension
- **Proper text preprocessing** with OpenCLIP tokenization
- **Embedding dimension matching** between training and inference
- **Grid and individual image saving options**

## 🔧 Troubleshooting History

### Major Issues Solved

**1. CLIP Model Mismatch**
- **Problem**: Training used ViT-L/14 laion2b_s32b_b82k, inference used wrong weights
- **Solution**: Updated inference script to auto-detect correct model from checkpoint
- **Symptom**: Cosine similarity -0.016 (should be ~0.3+)

**2. Training Instability**
- **Problem**: Discriminator overpowering generator, causing loss spikes
- **Solution**: Reduced learning rates, added gradient clipping, lower R1 penalty
- **Symptoms**: G loss >5, D loss near 0, mode collapse

**3. Memory Issues**
- **Problem**: CUDA OOM errors on T4 GPU
- **Solution**: Reduced batch size (8→4) and base channels (64→32)

**4. Tensor Size Mismatch**
- **Problem**: Fixed visualization samples had different batch size than training
- **Solution**: Added `actual_n_sample = min(args.n_sample, args.batch)` logic

## 📈 Training Progress Monitoring

### What to Watch For

**Good Signs:**
- D loss: 1.0-2.0 range, oscillating
- G loss: 0.5-1.5 range, decreasing slowly
- Real logits: Positive (0.5-1.5)
- Fake logits: Negative (-1.5 to 0)

**Bad Signs:**
- Loss spikes >5
- D loss dropping to near 0
- Real/fake logits going to extremes (±20)
- Generated samples becoming identical

### Recommended Stopping Points
- **25-30k steps**: Check quality improvement from baseline
- **40-50k steps**: Usually optimal quality/training time tradeoff
- **Stop immediately**: If losses explode or mode collapse occurs

## 💾 Output Structure

```
runs/cgan_768d/
├── ckpt_0010000.pt     # Model checkpoints every 10k steps
├── ckpt_0020000.pt
├── sample_0010000.png  # Generated samples every 10k steps
├── sample_0020000.png
├── training_log.json   # Loss curves data
└── training_plots.png  # Loss visualization plots
```

## 🔍 EMA Deep Dive

### Exponential Moving Average Theory

EMA is crucial for high-quality GAN training. Here's why and how it works:

### **Mathematical Foundation**
```python
# EMA update rule
θ_ema(t) = β * θ_ema(t-1) + (1-β) * θ_current(t)

# Where:
# β = decay rate (typically 0.999)
# θ_ema = EMA weights
# θ_current = current training weights
```

### **Why EMA Works**
1. **Training Noise**: Generator weights fluctuate during training due to:
   - Random mini-batch sampling
   - Gradient noise from stochastic optimization
   - Adversarial dynamics with discriminator

2. **Averaging Effect**: EMA creates a "smoothed" version of weights that:
   - Filters out high-frequency noise
   - Preserves long-term learning trends
   - Reduces visual artifacts in generated images

### **Implementation Flow in Training**
```python
# Training loop (simplified)
for step in range(training_steps):
    # 1. Forward pass with current weights
    fake_images = G(noise, text_embedding)

    # 2. Compute losses and gradients
    g_loss = adversarial_loss(fake_images)

    # 3. Update generator weights
    optimizer.step()

    # 4. Update EMA (key step!)
    ema.update(G)  # θ_ema = 0.999*θ_ema + 0.001*θ_current
```

### **EMA vs Regular Weights Comparison**
| Aspect | Training Weights | EMA Weights |
|--------|-----------------|-------------|
| **Stability** | Fluctuate with each update | Smooth, stable progression |
| **Noise** | High variance from gradient noise | Low variance, filtered |
| **Quality** | Can produce artifacts | Cleaner, more coherent outputs |
| **Training Use** | Used for gradient computation | Not used during training |
| **Inference Use** | Lower quality results | Higher quality results |

### **Decay Rate Impact**
```python
# Conservative (β = 0.999):
# Very stable, slow adaptation
# Good for final inference quality

# Moderate (β = 0.995):
# Balanced stability and adaptation
# Faster response to improvements

# Aggressive (β = 0.99):
# Less stable, quick adaptation
# May not filter enough noise
```

### **Visual Quality Differences**
**Regular Weights:**
- More noise and artifacts
- Inconsistent quality across batches
- Sharp edges may be jagged

**EMA Weights:**
- Smoother textures
- More consistent quality
- Better fine details
- Improved color coherence

### **Best Practices**
1. **Always use EMA for final inference**
2. **Start EMA after warmup period** (e.g., 1000 steps)
3. **Save EMA weights in checkpoints**
4. **Consider using EMA for training visualizations** (shows true model quality)
5. **Experiment with decay rates** based on dataset size and training stability

## 🔄 Workflow Summary

1. **Prepare data**: Enhanced CLIP embeddings (768D ViT-L/14)
2. **Train model**: Conservative hyperparameters for stability
3. **Monitor progress**: Check loss curves and sample quality
4. **Generate images**: Use text prompts with trained model
5. **Resume if needed**: From stable checkpoints with adjusted hyperparameters

The key insight was using higher-dimensional CLIP embeddings (768D vs 512D) for better text conditioning, combined with very conservative training hyperparameters to maintain stability.
