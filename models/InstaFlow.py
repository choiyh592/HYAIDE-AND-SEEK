import torch
import numpy as np
from PIL import Image
from einops import rearrange
from diffusers import StableDiffusionPipeline

class InstaFlow():
    def __init__(self, device="cuda"):
        self.device = device

        self.pipe = StableDiffusionPipeline.from_pretrained(
                "XCLiu/instaflow_0_9B_from_sd_1_5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        
        self.vae_scale = self.pipe.vae.config.scaling_factor
        self._zero_embeddings = self.prepare_zero_emb()
    
    def prepare_zero_emb(self):
        prompt = ""
        inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        locked_embeddings = self.pipe.text_encoder(inputs.input_ids)[0]
        zero_embeddings = torch.zeros_like(locked_embeddings).to(self.device)

        return zero_embeddings
    
    @property
    def zero_emb(self):
        return self._zero_embeddings

    @staticmethod
    def save_tensor_image_to_path(tensor_image, save_path):
        image = (tensor_image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, 'b c h w -> b h w c').cpu().float().numpy()
        image = (image[0] * 255).astype(np.uint8)
        
        Image.fromarray(image).save(save_path)
        
    def decode(self, x0):
        # Scale latents back up before VAE decoding
        x0_amplified = x0 / self.vae_scale
        recon_image = self.pipe.vae.decode(x0_amplified).sample
        return recon_image
    
    def encode(self, pixel_values):
        x1 = self.pipe.vae.encode(pixel_values).latent_dist.sample() * self.vae_scale
        return x1
    
    def velocity(self, x, t, embeddings):
        # InstaFlow(which is basically SD1.5) expects 0-1000 scale for timesteps
        t_in_sdformat = torch.tensor([t * 1000]).to(self.device, dtype=torch.float16)
        v = self.pipe.unet(x, t_in_sdformat, embeddings.to(self.device, dtype=torch.float16)).sample

        return v
    
    def process_image(self, init_image):
        image = self.pipe.image_processor.preprocess(init_image).to(self.device, dtype=torch.float16)
        image = image.to(self.device)
        return image