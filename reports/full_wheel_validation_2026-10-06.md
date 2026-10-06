# NeuronBridge Windows Wheel Full Validation Report

Date: 2026-10-06  
Version: `0.1.0a2`  
Python ABI: CPython 3.12 / Windows x86-64

## Release artifact

- Wheel: `D:\neuronBridge\artifacts\wheel\neuronbridge-0.1.0a2-cp312-cp312-win_amd64.whl`
- Size: 9,842,012 bytes
- SHA-256: `2D6732982E2899562EAF9286D866782BC9317D93D5B3CA09D23F2CC45C61CB52`
- Wheel inspection: PASS
- Isolated installation/runtime validation: PASS
- Bundled runtime coverage: CUDA runtime, ZeroMQ, Pinocchio and private MSVC/OpenMP runtime libraries present

## Results

| Area | Result | Coverage |
|---|---:|---|
| Python/API/release tests | PASS | 135 passed, 6 skipped, 2 non-interactive plotting warnings |
| ROKAE Arm with xMateSR3C URDF | PASS | 1 passed after explicitly supplying the local URDF |
| Native/model-codegen tests | PASS | 13/13 |
| CPU / legacy GPU / dense GPU | PASS | finite state, GPU readiness, CPU/GPU spike-time parity |
| Fast / streaming construction | PASS | build parity, weight mutation, save/load, reset preservation and cross-mode migration |
| Learning rules | PASS | STDP, R-STDP, AdditiveKernalChange, CerebullarLearningRule and three custom rules; fast and streaming paths |
| OuterDynamic | PASS | spike counter, planar arm and strict MATLAB planar arm; fast and streaming paths |
| DebugMonitor | PASS | main and dense/GPU state, spikes, weights and pending channels across seven neuron models |
| OpenMP accuracy | PASS | 1/2/4 queues; main network, four dense subnetworks, OuterDynamic and mixed network |
| Event queues | PASS | heap and timing wheel produce identical mixed-network results |
| Numerical accuracy | PASS | LIF decay, LIF double, Voltage Jump, Izhikevich and deterministic Poisson |
| Communication lifecycle | PASS | 10 publisher and synchronous REQ/REP server/client create-use-close cycles |
| Plotting | PASS | raster, state, weight, distribution, OuterDynamic and InputConv plots (10 outputs) |
| Realtime control | PASS | internal clock, EDLUT-style external clock, counters and non-finite input rejection |

## Skipped scope

Five data-dependent example cases were intentionally not run because they were previously excluded from scope:

- four handwriting datasets;
- `ei_connectivity` dataset.

The initially skipped ROKAE case was subsequently run with
`D:\Pinocchio\Model\xMateSR3C_description\urdf\xMateSR3C.urdf` and passed.

## Issues found and corrected during validation

1. The dense CUDA native smoke test still expected same-step interface firing. The runtime deliberately commits an InterfaceNeuron spike on the next dense update. The baseline now verifies an empty `t0` and neuron 0 firing at `t1`.
2. The new external realtime clock setter accepted `NaN` and infinity. The pybind boundary now rejects non-finite values with `ValueError`; the rebuilt wheel passes the new regression test.

## Detailed artifacts

- Wheel inspection: `D:\neuronBridge\artifacts\wheel\wheel-contents.json`
- Isolated runtime report: `D:\neuronBridge\build\wheel-validation-realtime-edlut\cuda-runtime-report.json`
- Full acceptance root: `D:\neuronBridge\artifacts\full_acceptance_realtime_edlut`
- Numerical accuracy: `D:\testfile\numerical_accuracy\numerical_accuracy_results.json`
- Extended regression: `D:\testfile\11_regression\extended_test_results.json`

## Notes

- The two warnings are expected `matplotlib` Agg-backend warnings from examples that call `plt.show()` in a headless test run; plot file generation itself passed.
- This report validates the Windows wheel. A Linux wheel must be built and tested on the target manylinux/Ubuntu build environment and is not represented by this artifact.
