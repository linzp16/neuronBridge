# C++ aligned 2-DOF cerebellum training

This project reproduces the configuration in
`D:/code_optimized/network_release/projects/cerebellum_framework/cerebellum_framework_arm_repro.cpp`
using the Python wheel.

The neural simulation runs with the native realtime controller. The external
2-DOF arm server communicates through asynchronous ZMQ PUB/SUB. The server
implements the same state-product encoding, CF error encoding, DCN torque
mapping, trajectory generation, and planar arm integration used by the C++
`StrictMatlabPlanarArm2DOFOuterDynamic` path.

Run from PowerShell:

```powershell
$env:PYTHONPATH = 'D:\neuronBridge\artifacts\realtime_test_env'
python train_cpp_aligned_realtime_async.py --output-dir D:\testfile\cerebellum_arm_realtime_async_cpp_aligned --epochs 3
```

The output directory contains per-epoch error reports, trajectory records, and
the final learned weight file.
