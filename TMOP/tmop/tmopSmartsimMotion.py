import torch
from jsonargparse import ArgumentParser, set_docstring_parse_options
import tmop

def main()->int:
    set_docstring_parse_options(attribute_docstrings=True)
    torch.set_default_dtype(torch.float64)
    parser = ArgumentParser(description="TMOP for mesh motion.")
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(tmop.MotionConfig, "motion")

    args = parser.parse_args()
    cfg: tmop.MotionConfig = args.motion

    solver = tmop.SmartSimMotionSolver(cfg)
    solver.solve()

    return 0

if __name__ == "__main__":
    main()
