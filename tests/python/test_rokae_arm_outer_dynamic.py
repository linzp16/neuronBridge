from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

import neuronbridge as nb


pytestmark = pytest.mark.skipif(
    not nb.backend_info()["native_extension_loaded"],
    reason="native extension is required",
)


def _reference_urdf() -> Path:
    configured = os.environ.get("NEURONBRIDGE_ROKAE_URDF", "")
    if not configured:
        pytest.skip("set NEURONBRIDGE_ROKAE_URDF to the xMateSR3C URDF")
    path = Path(configured)
    if not path.is_file():
        pytest.skip(f"ROKAE reference URDF does not exist: {path}")
    return path


def test_rokae_arm_six_dof_runtime_and_debug_monitor(tmp_path: Path):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_outer_dynamic(
        nb.OuterDynamic.rokae_arm(
            urdf_path=str(_reference_urdf()),
            name="rokae",
            trajectory_mode="hold",
            enable_pd_control=True,
        )
    )
    simulation = nb.Simulation(network, nb.SimulationConfig(steps=8, timestep=1.0))
    simulation.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=tmp_path,
            sample_interval_steps=1,
            flush_interval_steps=1,
            record_spikes=False,
            record_state=False,
            record_pending_channels=False,
            record_outer_dynamic_state=True,
        )
    )
    simulation.init()
    q = [0.0, 0.1, -0.2, 0.0, 0.15, 0.0]
    qd = [0.0] * 6
    simulation.reset_outer_dynamic_state("rokae", q, qd)
    simulation.set_outer_dynamic_desired_state("rokae", q, qd)
    simulation.run()

    state = simulation.outer_dynamic_state()
    for field in ("q", "qv", "qdd", "q_des", "qv_des", "tau_total"):
        assert len(state[field]) == 6
    assert state["q_des"] == pytest.approx(q)

    rows = list(csv.DictReader((tmp_path / "outer_dynamic_state.csv").open(encoding="utf-8")))
    q_indices = {int(row["index"]) for row in rows if row["field_name"] == "q"}
    assert q_indices == set(range(6))
