# About the TMOP motion solver
Motion solver for polyMesh based on the Target Matrix Optimization Paradigm.

The Python side of this solver can be configured via a `.yaml` config file. The default name for this file is `config.yaml`, which can be generated with its default values by running
```bash
python tmop_motion --print_config > config.yaml
```
in the `/run/meshMotion` directory.

> [!NOTE]
> To use a different config file you need to modify the script [`tmop_smartsim_driver.py`](../tmop_smartsim_driver.py):
> 1. in `tmop_rs` set `--config <your-config-name>.yaml`
> 2. in `tmop_motion.attach_generator_files()` include `"<your-config-name>.yaml"` in the `to_copy` list.

## Parameters
You can get help about parameters via
```
python tmop_run.py -h
```

### General parameters `cfg`
| name | type | description |
|------|------|-------------|
| mpi_ranks | int | Number of MPI ranks from OpenFOAM. Note: its value is overwritten by the SmartSim driver script. |
| device | str | Accelerator for PyTorch. Defaluts to 'cuda' if available, otherwise falls back to 'cpu'. |

### TMOP parameters `cfg.tmop`

| subclass | name | type | description |
|----------|------|------|-------------|
| shape | shape_fn | str | Name of the shape function on the reference elements. See [`shape_functions.py`](./shape_functions.py). |
| shape | regular_ref | bool | Whether the reference element should be a regular n-gon. Only supports `True` at the moment. |
| shape | n_samples | int | Number of sample points in the reference element. |
| metric | metric_fn | str | Name of the tmop metric to use. See [`tmopmetrics.py`](./tmopmetrics.py). |
| metric | gamma | float | Some metrics are a convex combination of others, e.g. m = (1-gamma) * m1 + gamma * m2. This parameter is the weight (should be in \[0,1\]).|
| metric | untangle | bool | Whether to force mesh untangling. Note that when this is `True` the metric should be 'non-barrier'.|
| metric | c | float | Parameter for untangling. Controls the offset of the barrier on the minimum value of the determinant.|
| metric | d | float | Parameter for worst case optimisation. Controls the offset of the barrier on the maximum value of the metric.|
| target | preserve_size | bool | Whether to preserve the size of the original mesh elements. |
| target | preserve_orientation | bool | Whether to preserve the orientation of the original mesh elements. |
| target | preserve_skewness | bool | Whether to preserve the skewness of the original mesh elements. If `False`, try to make elements as regular as possible.|
| target | preserve_aspect | bool | Whether to preserve the aspect ratio of the original mesh elements. If `False`, try to make elements as regular as possible.|
| optim | optimiser | torch.optim.Optim | Torch Optimizer (or subclass). To use custom optimizers, the corresponding module should be imported in [`tmop_motion.py`](../tmop_motion.py).|
| optim | lr | float | Learning rate. |
| optim | max_steps | int | Maximum number of optimisation epochs per timestep. |
| optim | patience | int | Maximum number of epochs without improvement. After `patience` epochs with no improvement, optimisation stops. |
| optim | rtol | float | Minimum relative tolerance to monitor improvement. A relative change in the maximum value of the loss greater than `rtol` is considered an improvement. |
