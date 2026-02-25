"""
이미지 생성을 엔지니어링 방식으로 시도 : SDXL Img2Img + IP-Adapter를 활용하여 conditional image generation
- Early frames: source-like
- Late frames: increasingly target-like
- Final frames: essentially target distribution (strength ~ 1.0)
"""

import os
from pathlib import Path
import math
from datetime import datetime
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLImg2ImgPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SOURCE_IMAGE_PATH = "image/image_6(tennis_balls, lemon).png"
TARGET_IMAGE_PATH = "image/image_6_target(tennis).jpg"
OUTPUT_DIR = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Keep prompt minimal so target image dominates
PROMPT = ""
NEGATIVE_PROMPT = ""

N_FRAMES = 20            # Total animation frames (including source and target)
BASE_STEPS = 35          # More steps finer results, but longer runtime. 
FINAL_STEPS = 55         # more steps for final convergence
BASE_GUIDANCE = 4.5      # High guidance : follow prompt more 
FINAL_GUIDANCE = 3.5     # lower guidance late so IP-adapter dominates more
SEED = 1234

TARGET_SIZE = (1024, 1024)
USE_MULTI_GPU = False

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Enable GPU runtime or install CUDA PyTorch.")
    print("✓ CUDA:", torch.cuda.get_device_name(0), "| GPUs:", torch.cuda.device_count())


def load_image_rgb(path: str, size=TARGET_SIZE):
    img = Image.open(path).convert("RGB")
    return img.resize(size, Image.Resampling.LANCZOS)


def setup_pipeline():
    require_cuda()
    kwargs = dict(torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.enable_vae_slicing() # converts pixel space to latent space (reduce memory)
    if not hasattr(pipe, "load_ip_adapter"):
        raise RuntimeError("diffusers has no IP-Adapter support. Run: pip install -U diffusers")
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT)
    pipe = pipe.to("cuda")
    print("✓ IP-Adapter loaded")
    return pipe


def strength_schedule(t: float) -> float:
    # 0.20 → 0.50 linearly
    return 0.20 + (0.50 - 0.20) * t


def ip_scale_schedule(t: float) -> float:
    # 0.70 → 1.10 linearly
    return 0.70 + (1.10 - 0.70) * t


def guidance_schedule(t: float) -> float:
    # 4.5 → 3.5 linearly (decreasing)
    return 4.5 + (3.5 - 4.5) * t


def steps_schedule(t: float) -> int:
    # 40 → 55 linearly
    return int(BASE_STEPS + (FINAL_STEPS - BASE_STEPS) * t)



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
    pipe = setup_pipeline()

    source = load_image_rgb(SOURCE_IMAGE_PATH)
    target = load_image_rgb(TARGET_IMAGE_PATH)
    target_ip = target.resize((224, 224), Image.Resampling.LANCZOS)

    source.save(os.path.join(OUTPUT_DIR, "source.png"))
    target.save(os.path.join(OUTPUT_DIR, "target.png"))

    frames = []

    for k in range(N_FRAMES):
        t = 0.0 if N_FRAMES == 1 else k / (N_FRAMES - 1)

        strength = float(strength_schedule(t))
        ip_s = float(ip_scale_schedule(t))
        guidance = float(guidance_schedule(t))
        steps = int(steps_schedule(t))

        pipe.set_ip_adapter_scale(ip_s)
        generator = torch.Generator(device="cuda").manual_seed(SEED)

        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=source,                 # always start from source to avoid noise accumulation
            ip_adapter_image=target_ip,   # target distribution (image)
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

        frames.append(out)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{k:03d}_t_{t:.2f}_strength_{strength:.2f}_ip_{ip_s:.2f}_guid_{guidance:.2f}_steps_{steps}.png"
        )
        out.save(out_path)
        print("Saved:", out_path)

    grid = make_grid(frames, cols=min(5, len(frames)))
    grid_path = os.path.join(OUTPUT_DIR, "grid_all_frames.png")
    grid.save(grid_path)
    print("Saved:", grid_path)

    save_strip(frames, OUTPUT_DIR)
    save_run_summary(OUTPUT_DIR, frames)

    print("\nDONE. Final frame should be very close to target distribution.")


if __name__ == "__main__":
    main()
