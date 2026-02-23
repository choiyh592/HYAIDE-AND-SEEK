import torch
from PIL import Image
from diffusers.utils import load_image


def image_generation(
    velocity_model,
    solver,
    image_path,
    alpha=0.5,
    num_steps_fw=1,
    num_steps_rev=10,
    init_time=0,
    target_size=1024,
):
    init_image = load_image(image_path).resize((target_size, target_size), Image.LANCZOS)

    with torch.no_grad():
        encoder = velocity_model.encoder
        decoder = velocity_model.decoder
        velocity_net = velocity_model.velocity
        processor = velocity_model.processor
        vae_scale = velocity_model.vae_scaling_factor
        tokenizer = velocity_model.tokenizer
        text_encoder = velocity_model.text_encoder

        pixel_values = processor.preprocess(init_image).to("cuda", dtype=torch.float16) * vae_scale
        x1 = encoder(pixel_values)

        prompt = ""
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to("cuda")
        locked_embeddings = text_encoder(inputs.input_ids)[0]
        locked_embeddings = torch.zeros_like(locked_embeddings)

        x_inv = x1.clone()
        x0 = solver(
            velocity_net,
            x_inv,
            locked_embeddings,
            num_steps_rev,
            mode="backward",
            init_time=init_time,
        )

        x0_amplified = (x0 + (x1 - x0) * init_time) * alpha
        x1_amplified = solver(
            velocity_net,
            x0_amplified,
            locked_embeddings,
            num_steps_fw,
            mode="forward",
            init_time=init_time,
        )
        x1_amplified = x1_amplified / vae_scale

        recon_image = decoder(x1_amplified)

    return init_image, recon_image
