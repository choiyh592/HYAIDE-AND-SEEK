import torch
import numpy as np
from PIL import Image
from diffusers import SanaPipeline
from einops import rearrange

class SANA():
    def __init__(self, device="cuda"):
        self.device = device

        self.pipe = SanaPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_1600M_512px_diffusers",
                variant="fp16",
                torch_dtype=torch.float16,
            ).to(device)
        
        self.vae_scale = self.pipe.vae.config.scaling_factor
        self._zero_embeddings = self.prepare_zero_emb()
    
    def prepare_zero_emb(self):
        prompt = ""
        self.inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
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
        x1 = self.pipe.vae.encode(pixel_values)
        return x1
    
    def velocity(self, x, step,num_inversion_step, embeddings):
        t_val = 1.0 - (step * num_inversion_step)
        t_tensor = torch.tensor([t_val], device="cuda", dtype=torch.float16)
        v = self.pipe.transformer(
                    hidden_states=x,
                    encoder_hidden_states=embeddings,
                    encoder_attention_mask=self.inputs.attention_mask, 
                    timestep=t_tensor,
                    return_dict=False
                )[0]
        return v
    
    def process_image(self, init_image):
        image = self.pipe.image_processor.preprocess(init_image).to(self.device, dtype=torch.float16)
        image = image.to(self.device)
        return image



    
