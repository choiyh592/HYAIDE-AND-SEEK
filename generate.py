import torch
from PIL import Image
from diffusers.utils import load_image

def image_generation(
    velocity_model,
    forward_solver,
    backward_solver,
    image_path,
    alpha=0.9,
    num_steps_fw=1,
    num_steps_rev=1,
    init_time=0,
    target_size=1024,
):
    init_image = load_image(image_path).resize((target_size, target_size), Image.LANCZOS)

    with torch.no_grad():
        pixel_values = velocity_model.process_image(init_image)
        x1 = velocity_model.encode(pixel_values)

        embeddings = velocity_model.zero_emb

        x_inv = x1.clone()
        x0 = backward_solver(
            velocity_model,
            x_inv,
            embeddings,
            num_steps_rev,
            mode="backward",
            init_time=init_time,
        )

        x0_amplified = (x0 + (x1 - x0) * init_time) * alpha
        x1_amplified = forward_solver(
            velocity_model,
            x0_amplified,
            embeddings,
            num_steps_fw,
            mode="forward",
            init_time=init_time,
        )

        recon_image = velocity_model.decode(x1_amplified)

    return init_image, recon_image
    