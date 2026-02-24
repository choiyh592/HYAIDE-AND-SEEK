# 단순한 VAE Latent Space Interpolation 

import os
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers.utils import load_image
from model import load_model


#MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_ID = "stabilityai/sdxl-turbo"
SOURCE_IMAGE_PATH = "image/image_8(labubu, apple).png"
TARGET_IMAGE_PATH = "image/image_8_target(labubu).jpg"
OUTPUT_DIR        = f"output_interp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

N_FRAMES      = 20
TARGET_SIZE   = 1024

def load_image_rgb(path: str, size=TARGET_SIZE):
    img = Image.open(path).convert("RGB")
    size = (size, size) if isinstance(size, int) else size
    return img.resize(size, Image.Resampling.LANCZOS)


def interpolate_images(
    velocity_model,
    solver,
    source_path,
    target_path,
    n_frames=N_FRAMES,
    target_size=TARGET_SIZE,
):
    src_img = load_image_rgb(source_path, size=target_size)
    tgt_img = load_image_rgb(target_path, size=target_size)

    with torch.no_grad():
        encoder      = velocity_model.encoder
        decoder      = velocity_model.decoder
        velocity_net = velocity_model.velocity
        processor    = velocity_model.processor
        vae_scale    = velocity_model.vae_scaling_factor
        tokenizer    = velocity_model.tokenizer
        text_encoder = velocity_model.text_encoder
        src_pixels = processor.preprocess(src_img).to("cuda", dtype=torch.float16) * vae_scale
        tgt_pixels = processor.preprocess(tgt_img).to("cuda", dtype=torch.float16) * vae_scale
        z_src = encoder(src_pixels)   # latent of source image  (x1 in vae.py terms)
        z_tgt = encoder(tgt_pixels)   # latent of target image

        prompt = ""
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to("cuda")
        locked_embeddings = text_encoder(inputs.input_ids)[0]
        locked_embeddings = torch.zeros_like(locked_embeddings)

        frames = []
        for k in range(n_frames):
            t = 0.0 if n_frames == 1 else k / (n_frames - 1)

            # Interpolate in latent space between source and target
            x1 = (1.0 - t) * z_src + t * z_tgt

            # Decode blended latent directly (no ODE, pure VAE decode)
            blend_image = decoder(x1)

            frames.append((t, blend_image))

    return src_img, tgt_img, frames

def tensor_to_pil(tensor):
    """Convert decoder output tensor (B, C, H, W) in [-1, 1] to PIL image."""
    img = (tensor / 2 + 0.5).clamp(0, 1)
    img = (img * 255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray(img)

def make_grid(images, cols=5):
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid

def save_strip(frames, out_dir):
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, frames)):
        ax.imshow(img)
        ax.set_title(f"t={i/(n-1):.2f}")
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "trajectory_strip.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("Saved:", path)
    plt.show()

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    velocity_model, solver = load_model(model_id=MODEL_ID, target_size=TARGET_SIZE)

    src_img, tgt_img, frames = interpolate_images(
        velocity_model=velocity_model,
        solver=solver,
        source_path=SOURCE_IMAGE_PATH,
        target_path=TARGET_IMAGE_PATH,
    )

    src_img.save(os.path.join(OUTPUT_DIR, "source.png"))
    tgt_img.save(os.path.join(OUTPUT_DIR, "target.png"))

    pil_frames = []
    blend_frames = []
    for k, (t, blend) in enumerate(frames):
        blend_pil = tensor_to_pil(blend) if isinstance(blend, torch.Tensor) else blend
        blend_path = os.path.join(OUTPUT_DIR, f"blend_{k:03d}_t_{t:.2f}.png")
        blend_pil.save(blend_path)
        print("Saved:", blend_path)
        blend_frames.append(blend_pil)


    blend_grid = make_grid(blend_frames, cols=min(5, len(blend_frames)))
    blend_grid.save(os.path.join(OUTPUT_DIR, "grid_blend_frames.png"))
    print("Saved:", os.path.join(OUTPUT_DIR, "grid_blend_frames.png"))

    save_strip(blend_frames, OUTPUT_DIR)

    print("\nDONE.")


if __name__ == "__main__":
    main()
