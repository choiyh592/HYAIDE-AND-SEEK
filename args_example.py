from pathlib import Path


def apply_local_defaults(parser):
    parser.set_defaults(
        image_path=Path("images/input/your_image.png"),
        save_path=Path("images/output"),
    )
