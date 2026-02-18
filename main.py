import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers.utils import load_image
import argparse
import tqdm

from solvers.ode_solvers import ODESolve
try:
    from args import apply_local_defaults
except ImportError:
    from args_example import apply_local_defaults

def image_generation(velocity_model, solver, image_path, alpha=0.5, 
                     num_steps_fw=1, num_steps_rev=10, init_time = 0, 
                     target_size=1024):
    # --- PREPROCESS ---
    init_image = load_image(image_path).resize((target_size, target_size), Image.LANCZOS)
    
    with torch.no_grad():
        # Retrieve Components
        encoder = velocity_model.encoder
        decoder = velocity_model.decoder
        velocity_net = velocity_model.velocity
        processor = velocity_model.processor
        vae_scale = velocity_model.vae_scaling_factor
        tokenizer = velocity_model.tokenizer
        text_encoder = velocity_model.text_encoder

        # Prepare Data
        pixel_values = processor.preprocess(init_image).to("cuda", dtype=torch.float16) * vae_scale
        x1 = encoder(pixel_values)

        # Prepare Embedding
        prompt = ""
        inputs = tokenizer(
            prompt, 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            return_tensors="pt"
        ).to("cuda")
        # Zero Embeddings
        locked_embeddings = text_encoder(inputs.input_ids)[0]
        locked_embeddings = torch.zeros_like(locked_embeddings)

        x_inv = x1.clone()
        
        x0 = solver(velocity_net, x_inv, locked_embeddings, num_steps_rev, mode='backward', init_time=init_time)
        
        x0_amplified = (x0 + (x1 - x0) * init_time ) * alpha
        
        x1_amplified = solver(velocity_net, x0_amplified, locked_embeddings, num_steps_rev, mode='forward', init_time=init_time)
        x1_amplified = x1_amplified / vae_scale

        recon_image = decoder(x1_amplified)
        
    return init_image, recon_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hidden Pictures Generation Script', add_help=False)

    # Required
    parser.add_argument('--image_path', type=Path, help='Path to original image')
    parser.add_argument('--save_path', type=Path, help='Path to save generated images')
    parser.add_argument('--model_id', type=str, required=True, help='model id (huggingface repo)')
    #TODO

    # Model specs
    #TODO

    apply_local_defaults(parser)
    args = parser.parse_args()
    model_id = args.model_id

    velocity_model = '' # TODO

    if args.method == 'euler':
        solver = ODESolve.euler_solver
    elif args.method == 'rk4':
        solver = ODESolve.rk4_solver

    orig, recon = image_generation(velocity_model, solver, args.image_path, alpha=args.alpha, 
                     num_steps_fw=args.num_steps_fw, num_steps_rev=args.num_steps_rev, init_time = args.init_time, 
                     target_size=args.target_size)
    
    args.save_path.mkdir(parents=True, exist_ok=True)
    orig.save(args.save_path / "instaflow_orig.png")
    recon.save(args.save_path / "instaflow_reconstruction.png")
