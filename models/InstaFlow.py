from typing import Self
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

class InstaFlow():
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
                "XCLiu/instaflow_0_9B_from_sd_1_5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to("cuda")
    
        self.velocity = self.pipe.unet
        self.processor = self.pipe.image_processor
        self.vae_scaling_factor = self.pipe.vae.config.scaling_factor

    def decoder(self, x0_amplified):
        reconstructed_latents = self.pipe(
            prompt="",
            num_inference_steps=1,
            guidance_scale=0.0,
            latents=x0_amplified
        ).images[0]

        return reconstructed_latents
    
    def encoder(self,pixel_values):
        x1 = self.pipe.vae.encode(pixel_values).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        return x1
    
    def tokenizer(self,prompt, padding, max_length, return_tensors):
        inputs = self.pipe.tokenizer(
            prompt, 
            padding, 
            max_length, 
            return_tensors
        ).to("cuda")

        return inputs
    
    def text_encoder(self, input):
        locked_embeddings = self.pipe.text_encoder(input)
        return locked_embeddings
