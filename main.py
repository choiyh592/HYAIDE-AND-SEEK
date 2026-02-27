import argparse
from pathlib import Path

from generate import image_generation
from interpret import run_interpret
from models.model_factory import FlowModels
from solvers.ode_solvers import ODESolve

def run_generate(args):
    if not args.model_id:
        raise ValueError("--model_id is required for generate task.")

    flow_wrapper = FlowModels(args)
    velocity_model = flow_wrapper.model
    forward_solver = ODESolve(args.ode_method_fw)
    backward_solver = ODESolve(args.ode_method_rev)

    orig, recon = image_generation(
        velocity_model,
        forward_solver,
        backward_solver,
        args.image_path,
        alpha=args.alpha,
        num_steps_fw=args.num_steps_fw,
        num_steps_rev=args.num_steps_rev,
        init_time=args.init_time,
        target_size=args.target_size,
    )

    args.save_path.mkdir(parents=True, exist_ok=True)
    orig.save(args.save_path / "instaflow_orig.png")
    velocity_model.save_tensor_image_to_path(recon, args.save_path / "instaflow_reconstruction.png")

def build_parser():
    parser = argparse.ArgumentParser("Hidden Pictures Pipeline")
    parser.add_argument("--task", type=str, choices=["generate", "interpret"], default="generate")
    parser.add_argument("--image_path", type=str, help="Path to original image")
    parser.add_argument("--save_path", type=Path, help="Path to save outputs")
    parser.add_argument("--model_id", type=str, help="Model id (huggingface repo)")
    parser.add_argument("--ode_method_fw", type=str, choices=["euler", "rk4"], default="euler", help="ODE solver method for forward solve")
    parser.add_argument("--ode_method_rev", type=str, choices=["euler", "rk4"], default="euler", help="ODE solver method for backwards solve")
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha value for latent scaling")
    parser.add_argument("--num_steps_fw", type=int, default=1, help="Number of forward ODE steps to use")
    parser.add_argument("--num_steps_rev", type=int, default=1, help="Number of backwards ODE steps to use")
    parser.add_argument("--init_time", type=float, default=0, help="Time to go backwards to")
    parser.add_argument("--target_size", type=int, default=1024, help="Target Image Size")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.task == "generate":
        run_generate(args)
    elif args.task == "interpret":
        run_interpret(args)

if __name__ == "__main__":
    main()
