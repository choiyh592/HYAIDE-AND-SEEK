import torch
import numpy as np
from PIL import Image
from diffusers import PixArtSigmaPipeline
from einops import rearrange

class PixartSigma():
    def __init__(self, device="cuda"):
        self.device = device

        self.pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            torch_dtype=torch.float16,
        ).to(device)

        
        self.vae_scale = self.pipe.vae.config.scaling_factor
        self._zero_embeddings = self.prepare_zero_emb()
    
    def prepare_zero_emb(self):
        max_length = 256 # 128-512 usually
        prompt = ""
        self.inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        locked_embeddings = self.pipe.text_encoder(self.inputs.input_ids)[0]
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
        x0_amplified = x0 / self.vae_scale
        recon_image = self.pipe.vae.decode(x0_amplified).sample
        return recon_image
    
    def encode(self, pixel_values):
        enc_output = self.pipe.vae.encode(pixel_values)

        if hasattr(enc_output, "latent_dist"):
            x1_raw = enc_output.latent_dist.sample()
        elif hasattr(enc_output, "latents"):
            x1_raw = enc_output.latents
        else:
            x1_raw = enc_output[0]

        x1 = x1_raw * self.vae_scale
        return x1
    
    def velocity(self, x, t, embeddings):
        # InstaFlow(which is basically SD1.5) expects 0-1000 scale for timesteps
        t_tensor = torch.tensor([t]).to(self.device)
        v = self.pipe.transformer(
                    hidden_states=x,
                    encoder_hidden_states=embeddings,
                    encoder_attention_mask=self.inputs.attention_mask, 
                    timestep=t_tensor,
                    return_dict=False
                )[0]

        # Pixart outputs two chunks for image and text (maybe not tho idk)
        # https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py#L185
        # we chunk the velocity and take the first chunk as in above implementation
        v = v.chunk(2, dim=1)[0]

        return v
    
    def process_image(self, init_image):
        image = self.pipe.image_processor.preprocess(init_image).to(self.device, dtype=torch.float16)
        image = image.to(self.device)
        return image    