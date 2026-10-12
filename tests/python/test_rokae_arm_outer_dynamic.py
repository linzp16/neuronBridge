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


def test_rokae_pd_off_target_error_does_not_generate_torque():
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_outer_dynamic(nb.OuterDynamic.rokae_arm(
        urdf_path=str(_reference_urdf()), name='no_pd', trajectory_mode='hold',
        enable_pd_control=False, pd_kp=[405.,405.,315.,20.,12.,8.],
        pd_kd=[10.,10.,8.,5.,3.,2.]))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=10,timestep=1.))
    sim.init()
    q = [0.,.1,-.2,0.,.15,0.]
    sim.reset_outer_dynamic_state('no_pd',q,[0.]*6)
    sim.set_outer_dynamic_desired_state('no_pd',[v+.2 for v in q],[.1]*6)
    sim.run()
    state=sim.outer_dynamic_state()
    assert state['tau_total'] == pytest.approx([0.]*6,abs=1e-12)


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


def test_rokae_arm_emits_six_joint_state_and_signed_error_feedback(tmp_path: Path):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(18))
    gc = [[joint] for joint in range(6)]
    cf_positive = [[6 + joint] for joint in range(6)]
    cf_negative = [[12 + joint] for joint in range(6)]
    network.add_outer_dynamic(
        nb.OuterDynamic.rokae_arm(
            urdf_path=str(_reference_urdf()),
            name="rokae_feedback",
            trajectory_mode="hold",
            enable_pd_control=True,
            update_timestep=2,
            communication_interval=2,
            state_feedback_repeats_per_update=1,
            error_feedback_window_steps=2,
            error_feedback_sample_count=1,
            spike_cf_max=10.0,
            cf_angle_norm_deg=[5.0] * 6,
            cf_velocity_norm_deg_s=[30.0] * 6,
            state_feedback_product=nb.FeedbackProductEncoding(
                bins=[1, 1, 1, 1],
                neuron_indices_by_joint=gc,
            ),
            cf_positive_neuron_indices_by_joint=cf_positive,
            cf_negative_neuron_indices_by_joint=cf_negative,
        )
    )
    simulation = nb.Simulation(network, nb.SimulationConfig(steps=12, timestep=1.0))
    simulation.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=tmp_path,
            sample_interval_steps=1,
            flush_interval_steps=1,
            record_spikes=True,
            record_state=False,
            record_pending_channels=False,
            record_outer_dynamic_state=False,
            neuron_ids=list(range(18)),
        )
    )
    simulation.init()
    initial = [0.0, 0.1, -0.2, 0.0, 0.15, 0.0]
    simulation.reset_outer_dynamic_state("rokae_feedback", initial, [0.0] * 6)
    simulation.set_outer_dynamic_desired_state(
        "rokae_feedback",
        [value + 0.1 for value in initial],
        [0.0] * 6,
    )
    simulation.run()

    rows = list(csv.DictReader((tmp_path / "spikes.csv").open(encoding="utf-8")))
    emitted = {int(row["global_neuron_id"]) for row in rows}
    assert set(range(6)).issubset(emitted)
    assert emitted.intersection(range(6, 18))
