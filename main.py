import argparse
from pathlib import Path

from generate import image_generation
from interpret import run_interpret
from models.model_factory import create_model
from solvers.ode_solvers import ODESolve

try:
    from args import apply_local_defaults
except ImportError:
    from args_example import apply_local_defaults


def run_generate(args):
    velocity_model = create_model(args)
    if velocity_model is None:
        raise NotImplementedError("create_model(args) is not implemented yet.")

    if args.method == "euler":
        solver = ODESolve.euler_solver
    else:
        solver = ODESolve.rk4_solver

    orig, recon = image_generation(
        velocity_model,
        solver,
        args.image_path,
        alpha=args.alpha,
        num_steps_fw=args.num_steps_fw,
        num_steps_rev=args.num_steps_rev,
        init_time=args.init_time,
        target_size=args.target_size,
    )

    args.save_path.mkdir(parents=True, exist_ok=True)
    orig.save(args.save_path / "instaflow_orig.png")
    recon.save(args.save_path / "instaflow_reconstruction.png")


def build_parser():
    parser = argparse.ArgumentParser("Hidden Pictures Pipeline")
    parser.add_argument("--task", type=str, choices=["generate", "interpret"], default="generate")
    parser.add_argument("--image_path", type=Path, help="Path to original image")
    parser.add_argument("--input_path", type=Path, help="Path to generated image")
    parser.add_argument("--save_path", type=Path, help="Path to save outputs")
    parser.add_argument("--model_id", type=str, required=True, help="Model id (huggingface repo)")
    parser.add_argument("--method", type=str, choices=["euler", "rk4"], default="euler")

    # Keep generation knobs as parser defaults without exposing CLI flags.
    parser.set_defaults(
        alpha=0.5,
        num_steps_fw=1,
        num_steps_rev=10,
        init_time=0.0,
        target_size=1024,
    )
    apply_local_defaults(parser)

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
