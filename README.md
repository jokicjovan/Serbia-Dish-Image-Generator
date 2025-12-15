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
  > "Sarma, a traditional Serbian dish of cabbage rolls served in a clay pot."
- No ingredient lists were included in the final caption; focus is on visual representation.

---

## 🧠 LoRA Fine-Tuned Stable Diffusion

### 📍 Base Model
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

### 📈 Training Dynamics

During training, several key metrics were monitored to ensure stable and effective learning:

#### Training Loss
The training loss was tracked throughout the process, showing the model's learning progress:

![Training Loss](docs/images/sd_lora/training_loss.png)

The graph shows three curves:
- **Raw Loss** (light blue): Individual batch losses showing natural variation
- **EMA Loss** (orange): Exponential Moving Average providing a smoothed trend
- **Rolling Average (100 steps)** (green): 100-step rolling average for medium-term trends

**Key observations:**
- Initial loss starts around 0.12 and quickly stabilizes
- EMA and rolling average converge around 0.16-0.17 after ~50 steps
- Relatively stable loss throughout training indicates good learning rate and proper convergence
- No signs of divergence or instability

#### Learning Rate Schedule
A linear learning rate warmup followed by cosine decay was employed:

![Learning Rate Schedule](docs/images/sd_lora/learning_rate.png)

**Schedule details:**
- **Warmup period:** 10 steps (0 to 0.0001)
- **Peak learning rate:** 0.0001 (1e-4)
- **Decay strategy:** Cosine annealing to near-zero
- **Final learning rate:** ~0 at step 500

This schedule ensures:
- Smooth initial adaptation (warmup)
- Stable training in the middle phase
- Fine-grained refinement toward the end (decay)

#### Gradient Norm
Gradient norms were monitored to detect potential training instabilities:

![Gradient Norm](docs/images/sd_lora/gradient_norm.png)

**Key observations:**
- Gradient norms remain consistently in the range of 0.004-0.012
- No gradient explosions or vanishing gradients observed
- Occasional spikes (e.g., around step 300) are normal and don't indicate problems
- Stable gradients throughout training suggest appropriate learning rate and good batch normalization

**Training stability:** The consistent gradient magnitudes across all 500+ steps demonstrate that the model learned effectively without encountering numerical instabilities, overfitting, or underfitting issues during this phase.

### 🖼️ Visual Examples — Before vs After LoRA

| Caption | Before LoRA | After LoRA |
|---------|-------------|------------|
| "Sarma, a traditional Serbian dish of cabbage rolls served in a clay pot." | ![Sarma base](docs/images/sd_lora/prompt_0_base.png) | ![Sarma lora](docs/images/sd_lora/prompt_0_lora.png) |
| "Ćevapi served with flatbread and onions." | ![Cevapi base](docs/images/sd_lora/prompt_1_base.png) | ![Cevapi lora](docs/images/sd_lora/prompt_1_lora.png) |

### 📊 Evaluation Metrics
Metrics are computed using captions and corresponding images from a subset of about 100 test samples, comparing generated images to ground-truth images with FID, CLIPScore, and CLIP cosine similarity to evaluate visual quality and semantic alignment.

#### Metrics Explained
- **FID (Fréchet Inception Distance):** measures distribution similarity between generated and real images (lower is better)
- **CLIPScore:** measures semantic alignment of caption and generated image on a 0-100 scale (higher is better)
- **CLIP cosine similarity:** measures alignment of text and generated image embeddings on a -1 to 1 scale (higher is better)

#### Results Across Training Steps

| Component | Training Step | FID ↓ | CLIPScore ↑ | CLIP cosine similarity ↑ |
|-----------|---------------|-------|-------------|--------------------------|
| UNet only | step 200 | 165.4340 | 64.52 | 0.2904 |
| UNet only | step 400 | 158.3018 | 64.45 | 0.2890 |
| UNet only | step 625 | 165.0374 | 64.61 | 0.2922 |
| UNet + Text Encoder | step 200 | 155.1722 | 64.68 | 0.2936 |
| UNet + Text Encoder | step 400 | 164.5307 | 64.42 | 0.2884 |
| UNet + Text Encoder | step 625 | 162.5785 | 64.37 | 0.2874 |

#### Metrics Visualization

| FID | CLIPScore | CLIP cosine similarity |
|-----|-----------|------------------------|
| ![FID](docs/images/sd_lora/fid_steps.png) | ![CLIP Score](docs/images/sd_lora/clip_score_steps.png) | ![CLIP cosine similarity](docs/images/sd_lora/clip_cosine_similarity_steps.png) |

#### Analysis & Conclusions

**UNet-only vs UNet + Text Encoder:**
- Training the Text Encoder alongside the UNet provides a **small performance boost in early training** (around step 200), improving both FID and text-image alignment
- As training continues beyond step 200, these gains **diminish and even reverse**, suggesting mild overfitting or instability when the Text Encoder is trained too long on a small dataset
- **UNet-only LoRA remains more stable** across steps, with less fluctuation in metrics

**Training duration observations:**
- All three evaluation checkpoints (200, 400, 625 steps) show relatively similar performance
- The model converges quickly, with most learning happening in the first 200 steps
- Extended training beyond 400 steps shows minimal improvement and may introduce slight degradation

**Recommendations:**
- For small datasets (~1,000 samples), training only the UNet provides **more stable and consistent results**
- Training the Text Encoder can be beneficial but should be **applied carefully with early stopping** (around step 200)
- Checkpoints around **step 200-400 offer the best balance** between performance and training stability

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

### 🧪 Additional Experiment: Extended Training on Larger Dataset

#### Training Configuration
In an additional experiment, LoRA was trained using:
- **Dataset size:** ~7,000 image-caption pairs (7× larger)
- **Training steps:** 1,000 steps (2× longer)
- **Fine-tuning target:** Both UNet and Text Encoder
- **Training duration:** Extended training phase for deeper adaptation

#### Quantitative Results
- **FID:** 113.2364 (**↓31.8% improvement** from ~165 baseline)
- **Average CLIPScore:** 64.13 (0-100 scale)
- **Average CLIP cosine similarity:** 0.2827 (-1 to 1 scale)

The **significant FID improvement** indicates that the generated image distribution is much closer to the real dataset distribution when trained on a larger dataset with more training steps.

#### Qualitative Analysis: Overfitting Observation

Despite improved quantitative metrics, **qualitative visual inspection reveals signs of overfitting**:

**Symptoms observed:**
- Generated images **closely replicate visual patterns, textures, and composition** from training data
- Dishes appear **too similar to specific training samples** rather than novel interpretations
- **Reduced visual diversity** when generating multiple images from the same prompt
- Loss of creative variation in plating, angles, and presentation

**Root causes:**
- The model learned **dataset-specific visual features too strongly**
- Extended training (1,000 steps) with both UNet and Text Encoder led to **memorization over generalization**
- Larger dataset provides more patterns to memorize, paradoxically increasing overfitting risk without proper regularization

#### Overfitting Example: Ćevapi

The following example illustrates overfitting behavior, where generated images strongly resemble specific training samples rather than producing novel variations:

| Caption | Generated Image |
|---------|-----------------|
| "Ćevapi served with flatbread and chopped onions." | ![Cevapi overfit](docs/images/sd_lora/cevapi_overfit.png) |

Notice how the generated image:
- Matches the exact plating style of training images
- Reproduces specific lighting and angle patterns
- Shows limited variation from known training examples

#### Key Takeaways

| Aspect | Finding | Recommendation |
|--------|---------|----------------|
| **Dataset Size** | Larger datasets improve FID but increase overfitting risk | Use datasets >5,000 images with proper regularization |
| **Training Duration** | 1,000 steps leads to memorization | Stop training around 400-600 steps for small-to-medium datasets |
| **Component Training** | Training both UNet + Text Encoder intensifies overfitting | Consider UNet-only for datasets <5,000 images |
| **Evaluation Approach** | Quantitative metrics alone are insufficient | **Always combine numerical metrics with qualitative visual inspection** |

**Conclusion:** While increasing dataset size and training duration improves FID scores, excessive fine-tuning—especially of both UNet and Text Encoder—can lead to overfitting. Visual realism improves at the cost of diversity and generalization. The **optimal training approach balances convergence with generalization**, requiring careful monitoring of both quantitative metrics and qualitative outputs.

---

## ⚡ CVAE Model
*Details to be added*

## ⚡ GAN Model
*Details to be added*
