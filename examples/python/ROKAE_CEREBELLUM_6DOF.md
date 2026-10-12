# ROKAE 6-DOF cerebellar training

## Control mode

Training examples now default to **neural-only control** (`pd_assist=False`).
The direct PD torque is disabled even when resuming an older PD-assisted run,
unless `--pd-assist` is explicitly provided to `tune_rokae_xy_circle.py`.
Desired position/velocity still supply GC state encoding and CF teaching errors;
they do not directly produce actuator torque. Gravity, passive joint damping,
URDF limits and torque limits remain part of the simulated plant.
The native model's general-purpose default is not changed; the examples
explicitly pass `enable_pd_control=False`.

### Pure neural experiment (100 epochs)

All training and replay stages explicitly disable PD. No inverse-dynamics or
gravity-compensation torque bypass is added. At each control update the motor
command is decoded DCN torque, subject to the existing actuator limits. Passive
plant damping and gravity are retained. Joint velocity and total actuator torque
are now included in `joint_tracking.csv` for auditing.

The `from_scratch` stage trained 40 epochs from random weights, followed by
60 epochs in `continue` with the same learning parameters. A separate `anneal`
branch tested a tenfold learning-rate reduction for the last 60 epochs.
The earlier PD-trained weights were tested separately with PD and learning off;
they are not ancestors of the selected pure-neural weights.

| Frozen replay | Endpoint RMSE (mm) | Maximum (mm) | Endpoint gap (mm) |
| --- | ---: | ---: | ---: |
| Untrained pure-neural network | 583.988 | 899.207 | 759.158 |
| Old PD-trained weights, PD switched off | 693.504 | 1116.698 | 1116.705 |
| Pure-neural 40 epochs | 52.898 | 149.249 | 148.784 |
| Pure-neural 100 epochs, reduced learning rate | 26.552 | 56.412 | 41.622 |
| Pure-neural 100 epochs, unchanged learning rate | 25.396 | 46.593 | 10.168 |

Selected online epoch 100: 24.767 mm RMSE; final five online epochs average
28.321 mm. Frozen replay is separate, with alpha=beta=0; two deterministic
replays matched and changed zero weights. This demonstrates improvement, not
precise circle convergence, unseen-path generalization or hardware readiness.

Selected parameters: 8x8x3x3 GC bins per joint, 96 neurons per signed PC/CF/DCN
population, alpha=0.001, beta=-0.0833333333, CF angle/velocity normalizers
5 deg / 45 deg/s, mixture 0.8, sample normalizer 50, spike_cf_max=7.5.
The decoded torque gains are `[0.0125,0.05,0.03125,0.015,0.01,0.003125]`.
PD gain values present in serialized configurations are inactive because
`pd_assist=false`; a regression verifies zero motor torque for nonzero desired
position/velocity errors when both PD and neural input are absent.

Reproduce from the repository root with the Python/native environment below,
using new output directories:

```powershell
& $pythonExe examples/python/tune_rokae_xy_circle.py --case local --nn 8 --nv 3 --population 96 --epochs 40 --output-dir artifacts/neural_repro/from_scratch
& $pythonExe examples/python/tune_rokae_xy_circle.py --case local --epochs 60 --resume artifacts/neural_repro/from_scratch --output-dir artifacts/neural_repro/continue
& $pythonExe examples/python/plot_rokae_end_effector.py --run-dir artifacts/neural_repro/continue/frozen
& $pythonExe examples/python/plot_rokae_circle_tuning.py --root artifacts/neural_repro --selected continue
```

Recorded outputs: `artifacts/rokae_neural_only/`; selected weights:
`artifacts/rokae_neural_only/continue/train/rokae_cerebellum_weights.txt`.
The plots read the saved control mode rather than labeling every run PD-assisted.
Scientific-toolkit separation of controls and Matplotlib visualization were used
to keep this result distinct from the historical hybrid experiments.
The 12 ROKAE/circle-example regression tests passed, including PD-off torque
isolation and the neural-only default. Python syntax and diff checks passed.

The following section documents a **historical PD-assisted experiment**, not
the performance of the neural-only controller. Its reproduction commands opt
into PD explicitly.

## Historical PD-assisted XY-circle tuning (2026-10-12)

The circle experiment is simulation-only and PD-assisted. It uses a 50 mm
radius, a 4 s target period, constant target height and orientation, and the
`world -> xMateSR3C_link6` URDF transform. The 201 control samples include both
endpoints; each sample runs a 20 ms control window (4.02 s simulated per epoch).
Each epoch resets joint state to the target start and preserves learned weights.
Actual endpoint positions are computed from logged actual joint angles, not
replaced by the desired circle. Curves are not smoothed or forced to close.

### What limited the original setup

- The default state code covered full URDF joint/velocity ranges with only
  3x3x2x2 bins per joint. The tuned code covers the target joint extrema plus
  10 degrees on each side, and velocity extrema times 1.5 (minimum 15 deg/s).
- Native CF count is rounded to an integer:
  `round(abs(spike_cf_max * n_cf / feedback_sample_count * mixed_error))`.
  Originally, with zero velocity error, errors below about 5.23 degrees
  produced zero teaching spikes. The tuned values reduce that threshold to
  about 0.217 degrees. They do not eliminate the quantization dead zone.
- Strengthening feedback alone destabilized the coarse network. Lowering
  learning rates alone did not resolve state-code ambiguity. More epochs
  were not always better: all tested candidates, including regressions,
  remain in `artifacts/rokae_xy_tune/comparison.csv`.

### Selected staged run

The retained run has 6,912 neurons, 663,552 plastic GC->PC connections and
1,329,408 neural connections, plus 1,152 plant output links. Each joint uses
576 GC neurons (8x8x3x3), and each signed PC/CF/DCN population uses 96 neurons.
Output connection gains were halved when doubling those populations, to
preserve the maximum population torque scale.

| Cumulative epochs | LTP alpha | LTD beta | PD Kp (first three joints) |
| --- | --- | --- | --- |
| 1-60 (`fine`) | 0.0001 | -0.0083333333 | 45, 45, 35 |
| 61-100 (`fine_pd`) | 0.00002 | -0.0016666667 | 135, 135, 105 |
| 101-140 (`stiff_pd`) | 0.00001 | -0.0008333333 | 405, 405, 315 |

Final remaining Kp values are 20, 12, 8; Kd remains 10, 10, 8, 5, 3, 2.
CF angle/velocity normalizers are 5 deg / 45 deg/s, position mixture is 0.8,
feedback sample normalization is fixed at 50, and `spike_cf_max=7.5`.
Torque gains are `[0.0125, 0.05, 0.03125, 0.015, 0.01, 0.003125]`.
Plastic bounds remain [1.5, 2.5], initial weights 2.2 plus uniform jitter
[0, 1/60]; initialization seed 7 and feedback/learning seed 17.
Exact per-joint encoding bounds are saved in every run's `train/config.json`.

| Measured evaluation | Endpoint RMSE (mm) | Maximum error (mm) | First/last sample gap (mm) |
| --- | ---: | ---: | ---: |
| Final online training epoch | 5.535 | 13.617 | 0.734 |
| Final weights, learning disabled | 5.817 | 13.873 | 9.017 |
| Same final PD/network parameters, untrained weights | 62.454 | 91.475 | 70.122 |

The final five online epochs average 5.889 mm RMSE. Two frozen replays from
the same initial conditions matched and changed zero weights. These are
deterministic repeatability checks, not independent random-seed or unseen-path
validation. Training and frozen results must not be conflated: this is a
substantial reduction in error, but not exact closed-circle tracking and not
proof of asymptotic convergence. The higher PD gains contribute to the result;
the learned controller is not operating without PD or tested on physical hardware.

### Reproduce in a new output directory

Use the local native build used for these experiments, not an older installed
wheel lacking the ROKAE feedback interface:

```powershell
$env:PYTHONPATH = 'D:\neuronBridge\build\wheel\python\Release'
$env:NEURONBRIDGE_ROKAE_URDF = 'D:\Pinocchio\Model\xMateSR3C_description\urdf\xMateSR3C.urdf'
$pythonExe = 'D:\anaconda\envs\neuronbridge-demo\python.exe'
& $pythonExe examples/python/tune_rokae_xy_circle.py --pd-assist --case local_slow --nn 8 --nv 3 --population 96 --epochs 60 --output-dir artifacts/rokae_circle_repro/fine
& $pythonExe examples/python/tune_rokae_xy_circle.py --pd-assist --case local_slow --epochs 40 --resume artifacts/rokae_circle_repro/fine --ltp-scale 0.2 --ltd-scale 0.2 --arm-kp-scale 3 --output-dir artifacts/rokae_circle_repro/fine_pd
& $pythonExe examples/python/tune_rokae_xy_circle.py --pd-assist --case local_slow --epochs 40 --resume artifacts/rokae_circle_repro/fine_pd --arm-kp-scale 3 --ltp-scale 0.5 --ltd-scale 0.5 --output-dir artifacts/rokae_circle_repro/stiff_pd
& $pythonExe examples/python/plot_rokae_end_effector.py --run-dir artifacts/rokae_circle_repro/stiff_pd/frozen
& $pythonExe examples/python/plot_rokae_circle_tuning.py --root artifacts/rokae_circle_repro --selected stiff_pd
```

Each stage also performs two frozen trained replays and two untrained controls.
`--resume` loads the preceding stage's topology/configuration and final weights;
topology changes are rejected on resume. Existing experiment directories are
rejected to avoid overwriting evidence. The selected weights are at
`artifacts/rokae_xy_tune/stiff_pd/train/rokae_cerebellum_weights.txt`.
Do not load these weights into the original smaller network.

The scientific-toolkit workflow was used to retain ablation results and explicit
units; Matplotlib plots include stage boundaries and unsmoothed frozen data.
NumPy 2.5.3 / Matplotlib 3.11.2 were used. Native ROKAE and circle-example
regressions passed (10 tests). An initial pytest attempt failed on a permissions
error in the system temp directory; a fresh `--basetemp` under `artifacts` fixed
the test environment without modifying or deleting that system directory.

`rokae_cerebellum_6dof.py` extends the existing two-joint cerebellar example to
all six xMate joints. Each joint owns one GC population and signed positive and
negative CF, PC, and DCN populations. All twelve GC-to-PC projections share the
same `CerebullarLearningRule` instance.

The native `ROKAE_Arm` model performs both feedback directions:

1. joint desired/actual position and velocity select one GC product-code neuron;
2. signed tracking error generates positive or negative CF spikes;
3. DCN positive-minus-negative activity is converted to joint torque.

Run a short smoke training:

```powershell
$env:NEURONBRIDGE_ROKAE_URDF = 'D:\Pinocchio\Model\xMateSR3C_description\urdf\xMateSR3C.urdf'
python examples/python/rokae_cerebellum_6dof.py --epochs 2 --samples-per-epoch 20
```

Run the default experiment:

```powershell
python examples/python/rokae_cerebellum_6dof.py --epochs 10 --samples-per-epoch 100
```

The defaults deliberately use a compact network for development. To approach
the original C++ scale, increase the product-code bins and output populations,
for example `--nn 15 --nv 10 --n-pc 200 --n-dcn 200 --gc-repeats 5`. For six
joints this is a very large network; validate memory requirements before using
that scale. The compact defaults are intended for API and control-loop checks.

Add `--debug-monitor` for a diagnostic run that records all network spikes and
the six-axis OuterDynamic state. This intentionally reduces performance and is
not recommended for long training runs.

Outputs are written to `artifacts/rokae_cerebellum_6dof`: per-epoch RMSE,
per-joint tracking traces, a training curve, plastic-weight change statistics,
and a reusable weight snapshot.
PD assistance is enabled by default. The tuning measurements below are for
this assisted simulated plant; `--no-pd-assist` is a separate experiment.

## Validated six-axis defaults (2026-10-12)

The script now defaults to 20 epochs and 100 samples/epoch (2 s each), with
48 neurons in each signed PC, CF and DCN population and 36 GC neurons/joint.
GC-PC weights start at 2.2 plus uniform [0, 0.016667), with bounds [1.5, 2.5].
GC-DCN excitation is 2.0; PC-DCN inhibition is 6.0. Five GC pulses are sent per
control window. Six per-spike torque weights are
`[0.025, 0.1, 0.0625, 0.03, 0.02, 0.00625]`.

```powershell
$env:PYTHONPATH = 'D:\neuronBridge\build\wheel\python\Release'
python examples/python/rokae_cerebellum_6dof.py --output-dir artifacts/rokae_training
python examples/python/rokae_cerebellum_6dof.py --disable-learning --output-dir artifacts/rokae_frozen
```

This requires the local native extension with ROKAE neural feedback support;
an older installed wheel may not have it. Each run saves `config.json` for
reproduction, metrics, weights and a plot. Equal PC/CF/DCN population sizes
are required by the one-to-one routing. When changing size, scale per-spike
torque gains inversely with population size before further tuning.

Matched 20-epoch tests gave last-five mean joint RMSE 0.182398 rad for the old
configuration and for frozen tuned weights, versus 0.058549 rad for learning
with the tuned configuration (67.9% lower). J1 remains slightly worse; this is
a single-seed repeated-trajectory result. Full per-joint data and sweep results:
`artifacts/rokae_tuning_report/report.md`.
