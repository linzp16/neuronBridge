# Cerebellum + external arm server

This project is based on `examples/python/cerebellum_framework.py`.
It removes the native `OuterDynamic` arm and replaces it with a Python arm
server connected through the asynchronous ZMQ spike protocol.

The control semantics are:

- the arm server advances one control phase whenever it receives a native
  output batch at the configured control interval;
- SNN output is sent with PUB/SUB and never blocks the simulation;
- the server publishes GC state feedback and CF signed error feedback through
  PUB/SUB;
- the native simulation uses `enable_realtime()` and `run_realtime()` so the
  control clock is real-time paced;
- learning weights are preserved between epochs and saved at the end.

All experiment settings are defined in the Python script. No external config
file is required.

Run with the rebuilt wheel environment:

```powershell
$env:PYTHONPATH='D:\neuronBridge\artifacts\realtime_test_env'
& 'D:\anaconda\envs\neuronbridge-demo\python.exe' `
  'D:\neuronBridge\cerebellum_arm_server_project\train_cerebellum_arm_server.py' `
  --output-dir 'D:\neuronBridge\artifacts\cerebellum_arm_server_project' `
  --epochs 3
```

The output includes per-epoch/per-phase reward, the received control steps,
the final arm trajectory, and the saved plastic weights.
