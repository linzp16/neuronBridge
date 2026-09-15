# Examples

The maintained Python examples live in `examples/python`. They use only the
public `neuronbridge` package and write generated output below
`artifacts/examples` by default.

## Start here

Run these after building the Release Python extension:

```powershell
$env:PYTHONPATH = "$PWD\build\windows-msvc-cuda\python\Release"
python examples\python\phase2_network_description.py
python examples\python\my_first_app.py
python examples\python\phase3_run_simulation.py
```

## Example groups

- Basic API: `phase2_network_description.py`, `my_first_app.py`, and
  `phase3_run_simulation.py`.
- Dense CUDA: `dense_subnetwork_export.py`, `dense_subnetwork_smoke.py`,
  `dense_mixed.py`, `dense_mixed_current.py`, and `dense_run_no_debug.py`.
- Visual input: `inputconv_poisson_dense.py` and `pattern_motion.py`.
- Communication: `zmqcommunication_client.py`; InputConv sync/async protocol
  coverage lives in `tests/python/test_migrated_examples.py`.
- Closed-loop simulation: `cerebellum_framework.py`.
- External-data applications: EI and handwriting stages 1 through 4.

Files beginning with `_` are shared implementation helpers, not standalone
entry points.

## Optional data

Large handwriting and EI datasets are not stored in the Git source tree. Set
`NEURONBRIDGE_DATA_ROOT` to the extracted data bundle root or place data under
`examples/data`. See `examples/data/README.md` and `manifest.json` for the
expected layout. Data-free smoke tests remain runnable without that bundle.

Numerical baselines and diagnostics belong under `tests`; generated results
belong under `artifacts` and are never committed.
