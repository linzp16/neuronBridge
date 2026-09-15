# NeuronBridge Visualization API Plan

## Purpose

This document defines the DebugMonitor visualization expansion. The native C++
runtime remains responsible for recording structured simulation data. Python owns
selection, aggregation, and Matplotlib rendering. This keeps visualization APIs
stable while allowing the runtime to evolve independently.

## Current Monitor Contracts

`weights.csv` records one value per sampled synapse:

```text
time_step,component_kind,component_index,component_name,synapse_id,value
```

`outer_dynamic_state.csv` currently records a single latest 2-DOF joint state:

```text
time_step,field_name,index,value
```

The current OuterDynamic fields are `q`, `qv`, `qdd`, `q_des`, `qv_des`, and
`tau_total`, with indices `0` and `1`.

## Phase 1: Python-only Visualization

Phase 1 does not alter the native extension or CSV schema. The following methods
are implemented on `DebugMonitorResult`.

```python
result.plot_weight_trace(component, synapse_ids, ax=None, limit=None)
result.plot_weight_summary(component, synapse_ids=None, statistics=("mean", "std"), band=True)
result.plot_weight_distribution(component, time_step=None, synapse_ids=None, bins=80)

result.plot_outer_dynamic_trace(fields=("q", "q_des"), indices=(0, 1), layout="overlay")
result.plot_outer_dynamic_tracking_error(field="q", desired_field=None, indices=(0, 1))
result.plot_outer_dynamic_phase_plane(joint_index=0, position_field="q", velocity_field="qv")
result.plot_outer_dynamic_torque(indices=(0, 1))
```

Weight APIs require a component name or `(component_kind, component_index)` pair
for normal usage. This prevents collisions between identically numbered synapses
in the main network and dense subnetworks. `plot_weight_summary()` aggregates
records incrementally by time step, avoiding a mandatory DataFrame load for large
monitor directories.

OuterDynamic plots are intentionally model-neutral: they render named recorded
fields and do not encode a robot kinematic model. A planar end-effector trajectory
plot belongs in a model-specific package or example because link lengths and joint
conventions are not part of the generic monitor contract.

All plotting methods accept a caller-owned Matplotlib `Axes` and return the
rendered axes. Methods create an explicit constrained-layout figure when no axes
is provided. They never call `show()` or write an image file.

## Phase 1 Validation

- Use compact fixture CSV files and Matplotlib's non-interactive `Agg` backend.
- Verify plotted line counts, labels, selected data values, summary bands, and
  histogram samples.
- Verify missing components, fields, synapse IDs, and invalid layout/statistic
  options fail with clear `ValueError` messages.
- Keep existing lazy `iter_rows()` and explicit `to_pandas()` behavior unchanged.

## Phase 2: Multi-OuterDynamic Native Schema

Implemented: the native collector stores the latest state for each reporting
OuterDynamic and writes the following schema:

```text
time_step,component_index,component_name,field_name,index,value
```

All OuterDynamic plot APIs accept an optional `outer_dynamic=` name or index
selector. The reader treats the old four-column format as the compatibility default
`component_index=0, component_name="outer_dynamic"`.

For live simulation inspection, `Simulation.outer_dynamic_states()` returns one
latest-state dictionary per reporting component. The older
`Simulation.outer_dynamic_state()` remains available and returns the most recently
written state for compatibility.

## Non-goals

- No C++ plotting code or GUI dependency.
- No implicit full-memory loading of large CSV files.
- No model-specific forward kinematics in the generic library.
- No API that exposes an individual C++ example executable as a special case.
