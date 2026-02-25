# test
from generate import image_generation
from models.model_factory import create_model
from solvers.ode_solvers import ODESolve
from pathlib import Path


class TempArgs:
    def __init__(self, model):
        self.model = model

if __name__ == "__main__":

    velocity_model = create_model(TempArgs("SANA"))
    forward_solver = ODESolve.euler_solver
    backward_solver = ODESolve.euler_solver

    image_path = "/home/yhchoi/Diffusion_Toy/corn_drawing.png"
    save_path = Path("/home/yhchoi/Diffusion_Toy")
    
    i, r = image_generation(velocity_model, forward_solver, backward_solver, image_path, alpha=0.8, init_time=0)

    i.save(save_path / "original.png")
    velocity_model.save_tensor_image_to_path(r, save_path / "recon.png")
    