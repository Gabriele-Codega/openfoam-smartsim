# Installation
In addition to the compilation and installation instructions in the root of this repository,
the mesh motion solvers used here need additional packages. First, install PhysicsNeMo

```
pip install nvidia-physicsnemo "Cython"
pip install nvidia-physicsnemo.sym --no-build-isolation
```

Then install the requirements here:

```
pip install -r requirements.txt
```

Find more details about the TMOP motion solver [here](./tmop/README.md).

> [!NOTE] 
> Due to current limitations in SmartRedis, you must use numpy < 1.26.4
