import torch
from jsonargparse import ArgumentParser, set_docstring_parse_options
import tmop

def main()->int:
    set_docstring_parse_options(attribute_docstrings=True)
    parser = ArgumentParser(description="TMOP for mesh motion.")
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(tmop.MotionConfig, "motion")

    args = parser.parse_args()
    cfg: tmop.MotionConfig = args.motion

    torch.set_default_dtype(cfg.dtype)
    solver = tmop.SmartSimMotionSolver(cfg)
    solver.solve()

    return 0

if __name__ == "__main__":
    main()
