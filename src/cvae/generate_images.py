import argparse
import os
import torch
import open_clip
import numpy as np
from model import ConvCVAE
from torchvision.utils import save_image
from torch import device,cuda

device = device("cuda" if cuda.is_available() else "cpu")
print(f"Using device: {device}")
def setup_clip_model(device):
    try:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=device
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        clip_model.eval()
        print("OpenCLIP model loaded successfully (ViT-B-32, laion2b_s34b_b79k)")
        return clip_model, tokenizer
    except Exception as e:
        print(f"Error loading OpenCLIP model: {e}")
        return None, None


def encode_text(text_prompt, clip_model, tokenizer, device):

    if clip_model is None or tokenizer is None:
        raise ValueError("OpenCLIP model not loaded.")

    if isinstance(text_prompt, str):
        text_prompt = [text_prompt]

    tokens = tokenizer(text_prompt).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().astype('float32')


def load_model(checkpoint_path, img_size, latent_dim, device):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Model not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    model = ConvCVAE(img_dim=[img_size, img_size, 3], latent_dim=latent_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def generate_images(model, text_prompt, clip_model, tokenizer, device,
                    num_samples=4, save_path="generated.png"):
    model.eval()

    print(f"\nPrompt: '{text_prompt}'")
    clip_embedding = encode_text(text_prompt, clip_model, tokenizer, device)
    clip_embedding = torch.from_numpy(clip_embedding).to(device)

    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        generated_images = model.generate(clip_embedding, num_samples=num_samples)

    generated_images = torch.clamp((generated_images * 0.5 + 0.5), 0, 1)

    nrow = min(4, num_samples)
    save_image(generated_images, save_path, nrow=nrow, padding=2)
    print(f"Saved {num_samples} generated images to: {save_path}")

    return generated_images



def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using trained CLIP-CVAE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default='best_checkpoint.pt',
                        help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Image size (must match training)")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="Latent dimension (must match training)")

    parser.add_argument("--prompt", type=str, default="AA rustic German bread loaf, golden-brown and crusty, with a hearty texture showcasing layers of rye and wheat flour, elegantly sliced and garnished with a sprinkle of flour, resting on a wooden board beside a bowl of warm, inviting butter.",
                             help="Single text prompt to generate from")

    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of variations to generate per prompt")
    parser.add_argument("--output", type=str, default="generated.png",
                        help="Output path for generated image(s)")
    parser.add_argument("--output_dir", type=str, default="generated_outputs",
                        help="Output directory when using --prompts_file")

    args = parser.parse_args()
    print("=" * 70)
    print("LOADING CLIP MODEL")
    print("=" * 70)
    clip_model, tokenizer = setup_clip_model(device)
    if clip_model is None:
        return

    # Load CVAE model
    print("\n" + "=" * 70)
    print("LOADING CVAE MODEL")
    print("=" * 70)
    model = load_model(args.checkpoint, args.img_size, args.latent_dim, device)

    # Generate images
    print("\n" + "=" * 70)
    print("GENERATING IMAGES")
    print("=" * 70)
    generate_images(
        model=model,
        text_prompt=args.prompt,
        clip_model=clip_model,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples,
        save_path=args.output
    )

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()