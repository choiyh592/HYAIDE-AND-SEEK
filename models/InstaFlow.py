import torch
from diffusers import StableDiffusionPipeline

class InstaFlow():
    def __init__(self, device="cuda"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
                "XCLiu/instaflow_0_9B_from_sd_1_5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        
        self.device = device

    @classmethod
    def decode(self, x0_amplified):
        reconstructed_latents = self.pipe(
            prompt="",
            num_inference_steps=1,
            guidance_scale=0.0,
            latents=x0_amplified
        ).images[0]

        reconstructed_latents = reconstructed_latents / self.pipe.vae.config.scaling_factor

        return reconstructed_latents
    
    @classmethod
    def encode(self, pixel_values):
        x1 = self.pipe.vae.encode(pixel_values).latent_dist.sample() * self.vae_scaling_factor
        return x1
    
    @classmethod
    def tokenize(self,prompt, padding, max_length, return_tensors):
        inputs = self.pipe.tokenizer(
            prompt, 
            padding, 
            max_length, 
            return_tensors
        ).to(self.device)

        return inputs
    
    @classmethod
    def encode_text(self, input):
        locked_embeddings = self.pipe.text_encoder(input)
        return locked_embeddings
    
    @classmethod
    def velocity(self, x, t, embeddings):
        # InstaFlow(which is basically SD1.5) expects 0-1000 scale for timesteps
        t_in_sdformat = torch.tensor([t * 1000]).to(self.device, dtype=torch.float16)
        v = self.pipe.unet(x, t_in_sdformat, embeddings.to(self.device, dtype=torch.float16)).sample

        return v
    
    @classmethod
    def process_image(self, init_image):
        image = self.pipe.image_processor.preprocess(init_image).to(self.device, dtype=torch.float16) * self.pipe.vae.config.scaling_factor
        return image