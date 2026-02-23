from marshal import version
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

class SDXL_Turbo():
    def __init__(self, image_path, target_size=512):
        self.target_size = target_size
        self.init_img = load_image(image_path).resize((target_size, target_size), Image.LANCZOS)

    def load_pipeline(self,pipe=None):
        if pipe is None:
            pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        self.pipe = pipe

    def invert_image(self, prompt="", num_inversion_steps=4):
        if not hasattr(self, "pipe"):
            raise RuntimeError("load_pipeline()을 먼저 호출하세요.")

        with torch.no_grad():
            # ----- VAE: image -> x1 (latent at t=1) -----
            pixel_values = self.pipe.image_processor.preprocess(self.init_img).to(device, dtype=torch.float16)
            self.x1 = self.pipe.vae.encode(pixel_values).latent_dist.sample() * self.pipe.vae.config.scaling_factor

            # ----- SDXL text embeddings (empty prompt, no CFG) -----
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            # Unconditional: use zeros like InstaFlow/SANA
            prompt_embeds = torch.zeros_like(prompt_embeds)
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

            # ----- SDXL added conditioning (resolution / crop) -----
            original_size = (self.target_size, self.target_size)
            crops_coords_top_left = (0, 0)
            target_size = (self.target_size,self.target_size)
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim
            add_time_ids = self.pipe._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_time_ids = add_time_ids.to(device).repeat(1, 1)
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            # ----- Inversion: x1 -> x0 (backward Euler, treat UNet output as velocity) -----
            x_inv = self.x1.clone()
            dt = 1.0 / num_inversion_steps
            for i in range(num_inversion_steps):
                t_val = 1.0 - (i * dt)
                # SDXL scheduler uses 0--999 scale
                t_tensor = torch.tensor([int(t_val * 1000)], device="cuda", dtype=torch.long)
                v = self.pipe.unet(
                    x_inv,
                    t_tensor,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                x_inv = x_inv - v * dt

            self.x0 = x_inv

    def sample(self,alpha,beta,gamma):
        if not hasattr(self, "x0") or not hasattr(self, "x1"):
            raise RuntimeError("invert_image()를 먼저 호출하세요.")
        noise = torch.rand_like(self.x0)
        # Discovery 식: InstaFlow/SANA와 동일 (alpha/beta/gamma 블렌드)
        x0_amplified = ((self.x0 + noise * gamma) * beta + self.x1 * (1 - beta)) * alpha
        reconstructed_latents = self.pipe(
            prompt="",
            num_inference_steps=1,
            guidance_scale=0.0,
            latents=x0_amplified,
            height=self.target_size,
            width=self.target_size,
            output_type="pil",
        ).images[0]

        recon_pt = self.pipe.image_processor.preprocess(reconstructed_latents)[0]
        orig_pt = self.pipe.image_processor.preprocess(self.init_img)[0]
        residual = torch.abs(orig_pt - recon_pt).mean(dim=0).cpu().numpy()
        
        
        res_map_normalized = (residual / (residual.max() + 1e-8) * 255).astype(np.uint8)
        self.init_img.save("sdxl_turbo_orig.png")
        Image.fromarray(res_map_normalized).save("sdxl_turbo_discovery.png")
        reconstructed_latents.save("sdxl_turbo_reconstruction.png")
