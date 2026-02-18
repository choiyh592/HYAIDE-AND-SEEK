from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image

def img_reshape(image, pixel_loc, target_size=224):
    orig_w, orig_h = image.size
    new_x = int(round(pixel_loc[0] * target_size / orig_w))
    new_y = int(round(pixel_loc[1] * target_size / orig_h))
    new_x = min(max(new_x, 0), target_size - 1)
    new_y = min(max(new_y, 0), target_size - 1)
    reshaped = image.resize((target_size, target_size), Image.BICUBIC)
    return reshaped, (new_x, new_y)


def get_attn(model, image_tensor, pixel_loc):
    attn_store = {}

    def _hook(_, __, output):
        attn_store["last_attn"] = output.detach()

    hook = model.blocks[-1].attn.attn_drop.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(image_tensor)
    finally:
        hook.remove()

    if "last_attn" not in attn_store:
        raise RuntimeError("Failed to capture attention tensor from last layer.")

    attn = attn_store["last_attn"]  # [B, heads, tokens, tokens]
    attn_mean = attn.mean(dim=1)[0]  # [tokens, tokens]

    patch = model.patch_embed.patch_size
    patch_h = patch[0] if isinstance(patch, tuple) else patch
    patch_w = patch[1] if isinstance(patch, tuple) else patch
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
    grid_h, grid_w = img_h // patch_h, img_w // patch_w

    px = min(max(pixel_loc[0] // patch_w, 0), grid_w - 1)
    py = min(max(pixel_loc[1] // patch_h, 0), grid_h - 1)
    query_idx = 1 + py * grid_w + px  # skip CLS token

    token_to_tokens = attn_mean[query_idx]  # [tokens]
    patch_tokens = token_to_tokens[1 : 1 + grid_h * grid_w]
    attn_map = patch_tokens.reshape(grid_h, grid_w)
    return attn_map.cpu().numpy()


def save_attn_map(attn_map, analysis_dir, image_name, image_idx):
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem
    npy_path = analysis_dir / f"{image_idx:04d}_{stem}_attn.npy"
    png_path = analysis_dir / f"{image_idx:04d}_{stem}_attn.png"

    np.save(npy_path, attn_map)

    norm = attn_map - attn_map.min()
    denom = norm.max() if norm.max() > 0 else 1.0
    norm = norm / denom
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    img = img.resize((224, 224), Image.NEAREST)
    img.save(png_path)
    return npy_path, png_path


def _extract_pixel_loc(df, idx):
    row = df.iloc[idx]
    cols = {c.lower(): c for c in df.columns}
    x_keys = ["x", "pixel_x", "px", "col", "x_loc", "xloc"]
    y_keys = ["y", "pixel_y", "py", "row", "y_loc", "yloc"]

    x_col = next((cols[k] for k in x_keys if k in cols), None)
    y_col = next((cols[k] for k in y_keys if k in cols), None)

    if x_col is not None and y_col is not None:
        return int(row[x_col]), int(row[y_col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        return int(row[numeric_cols[0]]), int(row[numeric_cols[1]])

    raise ValueError("Could not infer pixel location columns from item_loc.xlsx.")


def _load_image_from_output(output_dir, idx):
    output_dir = Path(output_dir)
    image_paths = sorted(
        p
        for p in output_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in output directory: {output_dir}")
    if idx < 0 or idx >= len(image_paths):
        raise IndexError(f"image_index={idx} is out of range (0..{len(image_paths)-1}).")
    return image_paths[idx]


def run_interpret(args):
    images_dir = Path("images")
    output_dir = Path(args.save_path) if args.save_path else images_dir / "output"
    analysis_dir = images_dir / "analysis"
    excel_path = images_dir / "item_loc.xlsx"
    image_idx = int(getattr(args, "image_index", 0))

    if not excel_path.exists():
        raise FileNotFoundError(f"Missing ROI file: {excel_path}")

    df = pd.read_excel(excel_path)
    if image_idx < 0 or image_idx >= len(df):
        raise IndexError(f"image_index={image_idx} is out of DataFrame range (0..{len(df)-1}).")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(analysis_dir / "item_loc_loaded.csv", index=False)

    image_path = _load_image_from_output(output_dir, image_idx)
    pixel_loc = _extract_pixel_loc(df, image_idx)

    image = Image.open(image_path).convert("RGB")
    reshaped, resized_pixel_loc = img_reshape(image, pixel_loc, target_size=224)

    arr = np.asarray(reshaped).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    model.eval()

    attn_map = get_attn(model, tensor, resized_pixel_loc)
    save_attn_map(attn_map, analysis_dir, image_path.name, image_idx)
