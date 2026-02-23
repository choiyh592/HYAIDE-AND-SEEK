import torch
import numpy as np
from PIL import Image
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

class Flux():
    def __init__(self,image_path,target_size):
        self.target_size = target_size
        self.init_img = load_image(image_path).resize((target_size, target_size), Image.LANCZOS)

    def load_pipeline(self,pipe):
        pass
    def invert_image(self,num_inversion_step,prompt=''):
        pass
    def sample(self,alpha,beta,gamma):
        pass