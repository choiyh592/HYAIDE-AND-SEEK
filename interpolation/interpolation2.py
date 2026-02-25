# VAE Latent Space Interpolation 하고 rectified flow 
"""
이미지 → VAE encoder → latent z
latent z → backward ODE → noise x0
noise interpolation
x0 → forward ODE → generated latent
generated latent → VAE decoder → image
"""

import os
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers.utils import load_image
from model import load_model

MODEL_ID = "stabilityai/sdxl-turbo"
SOURCE_IMAGE_PATH = "image/image_9(flowers, person).png"
TARGET_IMAGE_PATH = "image/image_9_target(flowers).jpg"
OUTPUT_DIR        = f"output_interp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"


N_FRAMES      = 20
ALPHA         = 1.0   # 0.5 was halving x0 before the forward pass, mis-matching the UNet's expected noise amplitude
NUM_STEPS_FW  = 4     # 1 caused a single dt=-14.6 Euler step that amplified any UNet error catastrophically
NUM_STEPS_REV = 4     # match forward steps so inversion and generation cover the same sigma range
INIT_TIME     = 0
TARGET_SIZE   = 1024

def _encode_ip_image(velocity_model, pil_image, device="cuda"):
    """
    Encode a PIL image into IP-Adapter image embeddings via the CLIP image encoder.
    Returns a list [tensor(1, 1, 1024)] ready for cross_attention_kwargs,
    or None if IP-Adapter is not loaded on the model.
    """
    fe  = velocity_model.feature_extractor
    enc = velocity_model.image_encoder
    if fe is None or enc is None:
        return None
    pixel_values = fe(images=pil_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        embeds = enc(pixel_values).image_embeds          # (1, 1024)
    return [embeds.unsqueeze(1)]                         # [(1, 1, 1024)]


def load_image_rgb(path: str, size=TARGET_SIZE):
    img = Image.open(path).convert("RGB")
    size = (size, size) if isinstance(size, int) else size
    return img.resize(size, Image.Resampling.LANCZOS)

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Enable GPU runtime or install CUDA PyTorch.")
    print("✓ CUDA:", torch.cuda.get_device_name(0), "| GPUs:", torch.cuda.device_count())



def setup_pipeline():
    require_cuda()
    kwargs = dict(torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    if USE_MULTI_GPU:
        kwargs["device_map"] = "balanced"
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL_ID, **kwargs)
    else:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.enable_vae_slicing() # converts pixel space to latent space (reduce memory)

    if not hasattr(pipe, "load_ip_adapter"):
        raise RuntimeError("diffusers has no IP-Adapter support. Run: pip install -U diffusers")

    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT)

    # Move to CUDA after IP adapter is loaded to ensure all components are on the same device
    if not USE_MULTI_GPU:
        pipe = pipe.to("cuda")

    print("✓ IP-Adapter loaded")
    return pipe


def interpolate_images(
    velocity_model,
    solver,
    source_path,
    target_path,
    n_frames=N_FRAMES,
    alpha=ALPHA,
    num_steps_fw=NUM_STEPS_FW,
    num_steps_rev=NUM_STEPS_REV,
    init_time=INIT_TIME,
    target_size=TARGET_SIZE,
):
    src_img = load_image_rgb(source_path, size=target_size)
    tgt_img = load_image_rgb(target_path, size=target_size)

    # Encode source and target into IP-Adapter embeddings (None if not loaded)
    ip_src = _encode_ip_image(velocity_model, src_img)
    ip_tgt = _encode_ip_image(velocity_model, tgt_img)

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

        # Invert source and target separately, each conditioned on its own image
        x0_src = solver(
            velocity_net,
            z_src,
            locked_embeddings,
            num_steps_rev,
            mode="backward",
            init_time=init_time,
            ip_adapter_embeds=ip_src,
        )
        x0_tgt = solver(
            velocity_net,
            z_tgt,
            locked_embeddings,
            num_steps_rev,
            mode="backward",
            init_time=init_time,
            ip_adapter_embeds=ip_tgt,
        )

        frames = []
        for k in range(n_frames):
            t = 0.0 if n_frames == 1 else k / (n_frames - 1)

            if t == 0.0:
                # Endpoint: decode source latent directly — no ODE drift
                blend_image = decoder(z_src)
            elif t == 1.0:
                # Endpoint: decode target latent directly — no ODE drift
                blend_image = decoder(z_tgt)
            else:
                # Step 1: Interpolate in noise space between inverted source and target
                x0_interp = (1.0 - t) * x0_src + t * x0_tgt

                # Step 2: Amplify — scale blended noise (with init_time=0 this is just x0_interp * alpha)
                x_clean_interp = (1.0 - t) * z_src + t * z_tgt
                x0_amplified = (x0_interp + (x_clean_interp - x0_interp) * init_time) * alpha

                # Step 3: Interpolate IP-Adapter conditioning between source and target image
                if ip_src is not None and ip_tgt is not None:
                    ip_interp = [(1.0 - t) * ip_src[0] + t * ip_tgt[0]]
                else:
                    ip_interp = None

                # Step 4: Forward ODE — generate from blended noise with blended image prompt
                x1_generated = solver(
                    velocity_net,
                    x0_amplified,
                    locked_embeddings,
                    num_steps_fw,
                    mode="forward",
                    init_time=init_time,
                    ip_adapter_embeds=ip_interp,
                )

                # Step 5: Decode to image
                blend_image = decoder(x1_generated)

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
