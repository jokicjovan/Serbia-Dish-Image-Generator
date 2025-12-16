# Serbian Dish Image Generator

This project aims to generate realistic images of Serbian dishes based on short text captions. Three models are being experimented with on the same dataset:
1. LoRA-fine-tuned Stable Diffusion
2. CVAE-based model (placeholder)
3. GAN-based model (placeholder)

The project is structured as follows:
- `data_pipeline/`: scripts for data collection, cleaning, preprocessing
  - `scraping/`: recipe scraping scripts from recepti.com and coolinarika.com
  - `preprocessing/`: image cleaning, cropping, resizing, and caption preparation
- `models_pipeline/`: scripts for training and evaluating models
  - `sd_lora/`: LoRA training and evaluation for Stable Diffusion
  - `cvae/`: placeholder for CVAE model
  - `gan/`: placeholder for GAN model

All models share the **same dataset** of Serbian dishes, consisting of ~1,000 image-caption pairs. Captions were generated using GPT-4o-mini based on scraped **dish name, ingredients, and preparation**, to produce a short description of dish appearance used as input for training and generation.

The dataset is split into **training and testing subsets** to allow evaluation and metric computation.

---

## 🗂 Data Collection & Preprocessing

### 1. Scraping
- Recipes were collected from **recepti.com** and **coolinarika.com** using custom scraping scripts in `data_pipeline/scraping/`.
- Data scraped included: **image, dish name, ingredients, preparation steps**.
- Only publicly available recipes were used.
- Images were downloaded and mapped to their corresponding dishes.

### 2. Cleaning & Cropping
- Duplicates, corrupt images, and irrelevant data were removed.
- Images were cropped/resized to square format (512×512) suitable for training.
- Scripts are located in `data_pipeline/preprocessing/`.

### 3. Caption Generation
- Captions were generated with GPT-4o-mini using the dish name, ingredients, and preparation steps.
- Each caption describes the dish appearance, e.g.:  
  > “Sarma, a traditional Serbian dish of cabbage rolls served in a clay pot.”
- No ingredient lists were included in the final caption; focus is on visual representation.

---

## 🧠 LoRA Fine-Tuned Stable Diffusion

### 🔍 Base Model
- **`runwayml/stable-diffusion-v1-5`**: strong generation quality, compatible with LoRA fine-tuning.

### 🧩 Why LoRA?
- Fine-tunes only a small fraction of weights (Low-Rank Adaptation).  
- Benefits:
  - Lightweight training and modular weight files  
  - Fast training on Google Colab T4 GPUs  
  - Avoids catastrophic forgetting of base SD knowledge  
  - Flexible training: **UNet-only** or **UNet + Text Encoder**

### 🔧 Training Setup
- **Device:** Google Colab T4 GPU  
- **Precision:** Mixed precision (fp16) via 🤗 Accelerate  
- **Memory optimization:** xFormers enabled if available  
- **Gradient accumulation:** supports small batches efficiently  
- **Checkpoints & weight saving:** UNet and Text Encoder LoRAs stored separately  
- **Training script location:** `models_pipeline/sd_lora/train_sd_lora.py`  
  - Can train only UNet, or UNet + Text Encoder, configurable via a flag in the script.

### 🧾 Dataset Input Format
- Each training example: **image + caption pair** (~1,000 total)  
- Captions are short descriptive sentences (no ingredient lists).  
- Dataset split: **training set** and **testing set** for evaluation  
- Guides the model to generate visually accurate representations.

### 🍽️ What LoRA Learns
| Component | Focus | Effect |
|-----------|-------|-------|
| UNet LoRA | Visual features: textures, plating, colors, presentation | Produces realistic Serbian food imagery |
| Text Encoder LoRA | Semantic understanding of dish captions | Better alignment of caption meaning to visual output |
| Combined | Full visual + semantic alignment | Strong, consistent generation results |

### 🖼️ Visual Examples – Before vs After LoRA

| Caption | Before LoRA | After LoRA |
|---------|-------------|------------|
| “Sarma, a traditional Serbian dish of cabbage rolls served in a clay pot.” | ![Sarma base](docs/images/sd_lora/prompt_0_base.png) | ![Sarma lora](docs/images/sd_lora/prompt_0_lora.png) |
| “Ćevapi served with flatbread and onions.” | ![Cevapi base](docs/images/sd_lora/prompt_1_base.png) | ![Cevapi lora](docs/images/sd_lora/prompt_1_lora.png) |

### 📊 Metrics
Metrics are computed using captions and corresponding images from a subset of about 100 test samples, comparing generated images to ground-truth images with FID, CLIPScore, and CLIP cosine similarity to evaluate visual quality and semantic alignment.
- **FID (Fréchet Inception Distance):** measures distribution similarity between generated and real images  
- **CLIPScore:** measures semantic alignment of caption and generated image  
- **CLIP cosine similarity:** measures alignment of text and generated image embeddings (-1 to 1 scale)

| Component | Training Step | FID | CLIPScore | CLIP cosine similarity |
|-----------|---------------|-----|-------|-------|
| UNet only | step 200 | 165.4340 | 64.52 | 0.2904 |
| UNet only | step 400 | 158.3018 | 64.45 | 0.2890 |
| UNet only | step 625 | 165.0374 | 64.61 | 0.2922 |
| UNet + Text Encoder | step 200 | 155.1722 | 64.68 | 0.2936 |
| UNet + Text Encoder | step 400 | 164.5307 | 64.42 | 0.2884 |
| UNet + Text Encoder | step 625 | 162.5785 | 64.37 | 0.2874 |

| FID | CLIPScore | CLIP cosine similarity |
|-----|-------|-------|
| ![Fid](docs/images/sd_lora/fid_steps.png) | ![Clip score](docs/images/sd_lora/clip_score_steps.png) | ![Clip cosine similarity](docs/images/sd_lora/clip_cosine_similarity_steps.png)

#### Conclusion
Fine-tuning the Text Encoder together with the UNet provides a small performance boost in the early training phase (around 200 steps), improving both FID and text-image alignment. However, as training continues, these gains diminish and even reverse, suggesting mild overfitting or instability when the Text Encoder is trained for too long on a small dataset. In contrast, UNet-only LoRA remains more stable across steps, with less fluctuation in metrics. Overall, training the Text Encoder can be beneficial, but should be applied carefully and with early stopping.

### 🚀 How to Use This LoRA
```python
from diffusers import StableDiffusionPipeline

base_model = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(base_model, safety_checker=None).to("cuda")

# Load LoRA weights (UNet or both)
pipe.load_lora_weights("models_pipeline/sd_lora/outputs/final/unet")
# pipe.load_lora_weights("models_pipeline/sd_lora/outputs/final/text_encoder")

prompt = "Sarma, a traditional Serbian dish of cabbage rolls served in a clay pot"
image = pipe(prompt).images[0]
image.save("output.png")
```

---

## ⚡ CVAE Model
### 🔍 Architecture Overview

- Conditional VAE: Generates and reconstructs images conditioned on CLIP text embeddings
- Encoder-Decoder Structure: Convolutional architecture with latent space bottleneck
- Embedding Dimension: 512-dimensional CLIP embeddings guide both encoding and generation

### 🧩 Why use a Conditional VAE (CVAE)?
- Pros

  - Stable and predictable training (no adversarial setup)
  - Explicit latent space enables smooth interpolation and controlled variation
  - Lightweight and suitable for training on consumer GPUs

- Cons
  - Generates blurrier images compared to GAN- or diffusion-based models
  - Limited scalability to high resolutions due to convolutional and memory constraints
  - Less capable of modeling fine-grained textures and high-frequency details

### 🔧 Training Setup

- Device: CUDA GPU (fallback to CPU)
- Optimizer: AdamW with weight decay (1e-5) for stable training
- Learning Rate: 2e-4 with ReduceLROnPlateau scheduler
- Loss Components:
  - Reconstruction Loss (MSE): ensures visual fidelity
  - KL Divergence: regularizes latent space distribution 
  - Beta-VAE weighting: configurable β parameter (default: 0.5)


- KL Warmup: gradual introduction of KL loss over first 10 epochs to prevent posterior collapse
Gradient Clipping: max norm 1.0 for training stability
Training script location: train.py
Model architecture: model.py (Encoder, Decoder, ConvCVAE)
Dataset handler: dataset.py (CaptionImageSet)

### 🧾 Dataset Input Format

- Each training example: image + pre-computed CLIP embedding (512-dim .npy file)
- Supported formats: .jpg, .jpeg, .png, .webp
- Dataset structure:
  - */images/: image files
  - */embeds/: corresponding .npy embedding files
- Data split: 90% training / 10% testing (configurable)
- Preprocessing: Resize to 64×64, normalize to [-1, 1] range
- CLIP embeddings are generated beforehand and guide the conditional generation process


### How to use this model

- Training from scratch
  - `python train.py \
    --data_root data/processed \
    --img_size 64 \
    --batch_size 128 \
    --latent_dim 128 \
    --beta 0.5 \
    --epochs 100 \
    --lr 2e-4`
- Resume from checkpoint
  - `python train.py --checkpoint path/to/checkpoint.pt`
- Generate image from existing checkpoint:
  - `python generate_images.py \
  --checkpoint path/to/best_checkpoint.pt \
  --prompt "Sarma, a traditional Serbian dish of cabbage rolls served in a clay pot" \
  --num_samples 8 \
  --output my_sarma.png`
  

## ⚡ GAN Model
*Details to be added*
