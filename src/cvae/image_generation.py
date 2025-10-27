import os
import open_clip
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from utils import convert_batch_to_image_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k',
        device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    print("OpenCLIP model loaded successfully (ViT-B-32, laion2b_s34b_b79k)")
except Exception as e:
    print(f"Error loading OpenCLIP model: {e}")
    print("Install with: pip install open_clip_torch")
    clip_model = None
    tokenizer = None


def encode_text(text_prompt):
    if clip_model is None or tokenizer is None:
        raise ValueError("OpenCLIP model not loaded")

    if isinstance(text_prompt, str):
        text_prompt = [text_prompt]

    tokens = tokenizer(text_prompt)
    tokens = tokens.to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().astype('float32')


def image_generation(model, target_attr, num_images=1, save_path=None):
    print(f"Generating {num_images} images with prompt: '{target_attr}'")

    text_embedding = encode_text(target_attr)

    condition = np.tile(text_embedding, (num_images, 1))

    condition_tf = tf.constant(condition, dtype=tf.float32)

    generated = model.generate(condition_tf)

    generated_np = generated.numpy()

    f = plt.figure(figsize=(12, 12))
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(convert_batch_to_image_grid(generated_np))
    plt.axis('off')

    prompt_str = str(target_attr).replace(' ', '_')
    plt.title(prompt_str, fontsize=20, pad=20)

    if save_path:
        if os.path.isdir(save_path):
            save_file = os.path.join(save_path, f"generation_{prompt_str}.png")
        else:
            save_file = save_path
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        print(f"Image saved to: {save_file}")

    plt.show()
    plt.close()

    return generated_np


def image_reconstruction(model, images, labels, save_path=None):
    batch_size = images.shape[0]
    img_size = images.shape[1]

    if batch_size != 32:
        if batch_size < 32:
            padding_size = 32 - batch_size
            images = np.concatenate([
                images,
                np.zeros((padding_size, img_size, img_size, 3), dtype=images.dtype)
            ], axis=0)
            labels = np.concatenate([
                labels,
                np.zeros((padding_size, labels.shape[1]), dtype=labels.dtype)
            ], axis=0)
        else:
            images = images[:32]
            labels = labels[:32]
    reconstructed = model.reconstruct(images, labels)
    reconstructed_np = reconstructed.numpy()

    f = plt.figure(figsize=(20, 10))

    ax = f.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(images))
    plt.title("Original Images", fontsize=20, pad=10)
    plt.axis('off')

    ax = f.add_subplot(1, 2, 2)
    ax.imshow(convert_batch_to_image_grid(reconstructed_np))
    plt.title("Reconstructed Images", fontsize=20, pad=10)
    plt.axis('off')

    if save_path:
        if os.path.isdir(save_path):
            save_file = os.path.join(save_path, "reconstruction.png")
        else:
            save_file = save_path
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        print(f"Image saved to: {save_file}")

    plt.show()
    plt.close()

    return reconstructed_np