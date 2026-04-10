# About the TMOP motion solver
Motion solver for polyMesh based on the Target Matrix Optimization Paradigm.

Install this package via
```bash
python -m pip install .
```
This will also create an executable `tmopSmarsimMotion`, which is somewhat similar to an OpenFOAM solver, and handles all the logic for mesh motion.

This solver can be configured via a `.yaml` config file. The default name for this file is `config.yaml`, which can be generated with its default values by running
```bash
tmopSmartsimMotion --print_config > config.yaml
```
in the `/run/meshMotion` directory.

> [!NOTE]
> To use a different config file you need to modify the script [`tmop_smartsim_driver.py`](../tmop_smartsim_driver.py):
> 1. in `tmop_rs` set `--config <your-config-name>.yaml`
> 2. in `tmop_motion.attach_generator_files()` include `"<your-config-name>.yaml"` in the `to_copy` list.

## Parameters
You can get help about parameters via
```bash
tmopSmartsimMotion -h
```

### General parameters `motion`
| name | type | description |
|------|:----:|-------------|
| mpi_ranks | int | Number of MPI ranks from OpenFOAM. Note: its value is overwritten by the SmartSim driver script. |
| device | str | Accelerator for PyTorch. Defaluts to 'cuda' if available, otherwise falls back to 'cpu'. |
| mode | str | One of `points0` and `incremental`. If `points0`, then at each time step the mesh is reset to its original configuration prior to TMOP optimisation. Otherwise the mesh from the previous time step is the starting point for optimisation.|
| tmop_optimiser | class | Either `tmop.TMOPOptimiser` or `tmop.WCUOptimiser`. The latter untangles meshes and tries to improve the worst quality element in the mesh. The former is standard TMOP. **Note**: the untangling optimiser should be used in conjuction with a non-barrier metric.|

### TMOP parameters `motion.tmop`

| subclass | name | type | description |
|----------|:----:|:----:|-------------|
| reference | shape_fn | str | Name of the shape function on the reference elements. See [`shape_functions.py`](./shape_functions.py). |
| reference | reference_type | str | Which shape should the reference element have. Must be one of `regular`, `initial`, and `svd`. |
| reference | n_samples | int | Number of sample points in the reference element. |
| metric | metric_fn | str | Name of the tmop metric to use. See [`tmopmetrics.py`](./tmopmetrics.py). |
| metric | gamma | float | Some metrics are a convex combination of others, e.g. m = (1-gamma) * m1 + gamma * m2. This parameter is the weight (should be in \[0,1\]).|
| metric | c | float | Parameter for untangling. Controls the offset of the barrier on the minimum value of the determinant.|
| metric | d | float | Parameter for worst case optimisation. Controls the offset of the barrier on the maximum value of the metric.|
| target | target_factory | class | Either `TargetInitial` or `TargetIdentity`, Determines how the target matrices are built. `TargetInitial` uses as target matrices the jacobian of the transformation from reference to initial mesh. `TargetIdentity` uses the identity matrix as target, which means that the elements will become as similar as possible to the reference element.|
| optim | optimiser | torch.optim.Optim | Torch Optimizer (or subclass). |
| optim | optimiser_kwargs | dict | Keyword arguments for the optimiser. |
| optim | scheduler | torch.optim.lr_scheduler.LRScheduler | Torch LRScheduler (or subclass). |
| optim | scheduler_kwargs | dict | Keyword arguments for the learning rate scheduler. |
| optim | t_scheduler | torch.optim.lr_scheduler.LRScheduler | Torch LRScheduler (or subclass). This scheduler can be used to adapt the learning rate during the untangling phase. The prefix `t_` indicates that it will try to monitor the value of `t`, that is the minimum determinant of the active jacobians (negative for inverted elements). |
| optim | t_scheduler_kwargs | dict | Keyword arguments for the `t_` learning rate scheduler. |
| optim | max_steps | int | Maximum number of optimisation epochs per timestep. |
| optim | patience | int | Maximum number of epochs without improvement. After `patience` epochs with no improvement, optimisation stops. |
| optim | rtol | float | Minimum relative tolerance to monitor improvement. A relative change in the maximum value of the loss greater than `rtol` is considered an improvement. |
| optim | batch_size | int | Batch size. Set to `-1` to disable batching. **Note: batching is done over mesh points, and it does not seem beneficial. This feature is here because... idk, you never know ¯\\\_(ツ)\_/¯**. |
