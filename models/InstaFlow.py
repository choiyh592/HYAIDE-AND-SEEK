
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

class InstaFlow():
    def __init__(self, image_path, target_size=1024):
        self.init_img = load_image(image_path).resize((target_size, target_size), Image.LANCZOS)
    
    def load_pipeline(self, pipe=None):
        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                "XCLiu/instaflow_0_9B_from_sd_1_5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to("cuda")
        self.pipe = pipe

    def invert_image(self,prompt=""):
        if not hasattr(self, "pipe"):
            raise RuntimeError("load_pipeline()을 먼저 호출하세요.")
        # ----- Encoding image to Latents (x1) -----
        with torch.no_grad():
            pixel_values = self.pipe.image_processor.preprocess(self.init_img).to("cuda", dtype=torch.float16)
            
            self.x1 = self.pipe.vae.encode(pixel_values).latent_dist.sample() * self.pipe.vae.config.scaling_factor
            steps = 1
            
            inputs = self.pipe.tokenizer(
                prompt, 
                padding="max_length", 
                max_length=self.pipe.tokenizer.model_max_length, 
                return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                # Capture the exact hidden states
                locked_embeddings = self.pipe.text_encoder(inputs.input_ids)[0]
                locked_embeddings = torch.zeros_like(locked_embeddings)
            # ----- Encoding Finished -----
            
            #----- invert x1 to x0 -----
            ### should be fixed (including ode solver)
            x_inv = self.x1.clone()
            dt = 1.0 / steps
            for i in range(steps):
                t = 1.0 - (i * dt)

                # InstaFlow/SD1.5 expects 0-1000 scale for timesteps
                t_tensor = torch.tensor([t * 1000]).to("cuda", dtype=torch.float16)
                v = self.pipe.unet(x_inv, t_tensor, locked_embeddings.to("cuda", dtype=torch.float16)).sample
                x_inv = x_inv - v * dt

            self.x0 = x_inv
            
    def sample(self,alpha,beta,gamma=0.1):
        if not hasattr(self, "x0") or not hasattr(self, "x1"):
            raise RuntimeError("invert_image()를 먼저 호출하세요.")
        noise = torch.rand_like(self.x0)
        x0_amplified = ((self.x0 + noise * gamma) * beta + self.x1 * (1-beta)) * alpha
        reconstructed_latents = self.pipe(
            prompt="",
            num_inference_steps=1,
            guidance_scale=0.0,
            latents=x0_amplified
        ).images[0]

        recon_pt = self.pipe.image_processor.preprocess(reconstructed_latents)[0]
        orig_pt = self.pipe.image_processor.preprocess(self.init_img)[0]
        residual = torch.abs(orig_pt - recon_pt).mean(dim=0).cpu().numpy()
        
        
        res_map_normalized = (residual / (residual.max() + 1e-8) * 255).astype(np.uint8)
        self.init_img.save("instaflow_orig.png")
        Image.fromarray(res_map_normalized).save("instaflow_discovery.png")
        reconstructed_latents.save("instaflow_reconstruction.png")