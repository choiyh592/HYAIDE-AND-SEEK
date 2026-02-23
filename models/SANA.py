import torch
import numpy as np
from PIL import Image
from diffusers import SanaPipeline
from diffusers.utils import load_image

class SANA():
    def __init__(self, image_path, target_size=512):
        self.target_size = target_size
        self.init_img = load_image(image_path).resize((self.target_size, self.target_size), Image.LANCZOS)
    
    def load_pipeline(self, pipe=None):
        if pipe is None:
            pipe = SanaPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_1600M_512px_diffusers",
                variant="fp16",
                torch_dtype=torch.float16,
            ).to("cuda")
        self.pipe = pipe

    def invert_image(self, num_inversion_steps, prompt=""):
        if not hasattr(self, "pipe"):
            raise RuntimeError("load_pipeline()을 먼저 호출하세요.")
        
        with torch.no_grad():
            # --- VAE ENCODING ---
            pixel_values = self.pipe.image_processor.preprocess(self.init_img).to("cuda", dtype=torch.float16)
            enc_output = self.pipe.vae.encode(pixel_values)
            
            # Extract and scale
            x1_raw = enc_output.latents if hasattr(enc_output, "latents") else enc_output[0]
            self.x1 = x1_raw * self.pipe.vae.config.scaling_factor
            # --- TEXT EMBEDDINGS ---
            max_length = 256 
            prompt_outputs = self.pipe.tokenizer(
                prompt, padding="max_length", max_length=max_length, 
                truncation=True, return_tensors="pt"
            ).to("cuda")
            prompt_embeds = self.pipe.text_encoder(
            prompt_outputs.input_ids, 
            attention_mask=prompt_outputs.attention_mask
            )[0]
            self.locked_embeddings = torch.zeros_like(prompt_embeds)

        # --- inversion ---
            x_inv = self.x1.clone()
            dt = 1.0 / num_inversion_steps
            
            for i in range(num_inversion_steps):
                # We go from t=1.0 down to 0.0
                # Step i=0: t=1.0, Step i=num_steps-1: t=dt
                t_val = 1.0 - (i * dt)
                t_tensor = torch.tensor([t_val], device="cuda", dtype=torch.float16)
                
                # Predict velocity at current state and time
                v = self.pipe.transformer(
                    hidden_states=x_inv,
                    encoder_hidden_states=self.locked_embeddings,
                    encoder_attention_mask=prompt_outputs.attention_mask, 
                    timestep=t_tensor,
                    return_dict=False
                )[0]
                
                # Backward Euler step: x_{t-dt} = x_t - v * dt
                x_inv = x_inv - v * dt
            
            self.x0 = x_inv

    def sample(self, alpha, beta,gamma=0.1):
        if not hasattr(self, "x0") or not hasattr(self, "x1"):
            raise RuntimeError("invert_image()를 먼저 호출하세요.")

        noise = torch.randn_like(self.x0)
        x0_manipulated = ((self.x_inv)* beta + self.x1 * (1-beta)) * alpha
        
        # --- RECONSTRUCTION ---
        # Note: num_inference_steps=1 works for SANA's flow-matching
        output = self.pipe(
            prompt="",
            num_inference_steps=2,
            guidance_scale=1.0, 
            latents=x0_manipulated,
            width=self.target_size,   # Explicitly set width/height
            height=self.target_size,
            output_type="pil"
        ).images[0]
        
        # --- RESIDUAL CALCULATION (Tensor Alignment) ---
        recon_pt = self.pipe.image_processor.preprocess(output).to("cuda")
        orig_pt = self.pipe.image_processor.preprocess(self.init_img).to("cuda")
        
        # Safety check: if shapes still differ (due to padding/VAE logic), resize recon to match orig
        if recon_pt.shape != orig_pt.shape:
            recon_pt = torch.nn.functional.interpolate(recon_pt, size=(self.target_size, self.target_size), mode='bilinear')

        residual = torch.abs(orig_pt - recon_pt).mean(dim=1).squeeze().cpu().numpy()
        

        res_map_normalized = (residual/ (residual.max() + 1e-8) * 255).astype(np.uint8)
        self.init_img.save("sana_orig.png")
        Image.fromarray(res_map_normalized).save("sana_discovery.png")
        output.save("sana_reconstruction.png")
        print("Discovery maps saved successfully.")

    