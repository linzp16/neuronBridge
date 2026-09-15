"""Python migration for the C++ cerebellum_framework example.

The example keeps the C++ topology: GC/CF input populations, PC/DCN LIF
populations, a cerebellar learning rule, and a 2-DOF arm OuterDynamic. Defaults
are intentionally small so the example is useful as a package smoke test; pass
larger sizes to approach the C++ project configuration.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import neuronbridge as nb
from _example_paths import artifact_dir


CONDUCTANCE_SCALE = 1.0 / 6.0


@dataclass(frozen=True)
class CerebellumConfig:
    sample_count: int = 6
    dt_ms: float = 0.1
    control_interval_ms: float = 20.0
    nn: int = 3
    nv: int = 2
    n_pc: int = 12
    n_cf: int = 12
    n_dcn: int = 12
    gc_repeats_per_control: int = 2
    link1: float = 0.34
    link2: float = 0.32
    mass1: float = 1.8
    mass2: float = 1.6
    w_gc_dcn: float = 12.0 * CONDUCTANCE_SCALE
    w_pc_dcn: float = 12.0 * CONDUCTANCE_SCALE
    w_gcpc_init: float = 11.0 * CONDUCTANCE_SCALE
    w_gcpc_jitter: float = 0.1 * CONDUCTANCE_SCALE
    w_gcpc_max: float = 12.0 * CONDUCTANCE_SCALE
    w_gcpc_min: float = 10.0 * CONDUCTANCE_SCALE
    alpha: float = 6.0e-3 * CONDUCTANCE_SCALE
    beta: float = -0.5 * CONDUCTANCE_SCALE
    dcn_torque_gain_1: float = 0.2
    dcn_torque_gain_2: float = 0.05
    cf_mix_position: float = 0.8
    spike_cf_max: float = 15.0

    @property
    def n_gc(self) -> int:
        return self.nn * self.nn * self.nv * self.nv

    @property
    def steps_per_control(self) -> int:
        return int(round(self.control_interval_ms / self.dt_ms))

    @property
    def total_steps(self) -> int:
        return self.steps_per_control * self.sample_count


@dataclass(frozen=True)
class LayerIndex:
    gc1: int
    gc2: int
    cf1: int
    cf2: int
    cf3: int
    cf4: int
    pc1: int
    pc2: int
    pc3: int
    pc4: int
    dcn1: int
    dcn2: int
    dcn3: int
    dcn4: int


@dataclass(frozen=True)
class StateSample:
    q1: float
    q2: float
    qd1: float
    qd2: float
    x: float
    y: float


def _lif_layer(count: int, *, output: bool = False) -> nb.NeuronLayer:
    return nb.NeuronLayer.lif_double(
        count,
        output=output,
        V_rest=nb.float32(-70.0),
        V_reset=nb.float32(-80.0),
        V_th=nb.float32(-40.0),
        tau=nb.float32(10.0 / 6.0),
        R=nb.float32(1.0),
        t_ref=nb.int32(10),
        gexc_tau=nb.float32(1.0),
        ginh_tau=nb.float32(1.0),
        Eexc=nb.float32(0.0),
        Einh=nb.float32(-80.0),
        random_mu=nb.float32_array4([-10.0, 0.0, 0.0, 0.0]),
        random_sigma=nb.float32_array4([0.0, 0.0, 0.0, 0.0]),
    )


def add_layers(network: nb.Network, cfg: CerebellumConfig) -> LayerIndex:
    starts: list[int] = []
    next_id = 0
    for count in [cfg.n_gc, cfg.n_gc, cfg.n_cf, cfg.n_cf, cfg.n_cf, cfg.n_cf]:
        starts.append(next_id)
        next_id += count
        network.add_layer(nb.NeuronLayer.input_spike(count))
    for _ in range(4):
        starts.append(next_id)
        next_id += cfg.n_pc
        network.add_layer(_lif_layer(cfg.n_pc))
    for _ in range(4):
        starts.append(next_id)
        next_id += cfg.n_dcn
        network.add_layer(_lif_layer(cfg.n_dcn, output=True))
    return LayerIndex(*starts)


def _dense_projection(source_begin: int, source_count: int, target_begin: int, target_count: int, synapse_type: int, weight: float):
    return nb.Connection(
        source=[source_begin + source for target in range(target_count) for source in range(source_count)],
        target=[target_begin + target for target in range(target_count) for source in range(source_count)],
        synapse_type=synapse_type,
        weight=weight,
        max_weight=weight,
        delay=1,
    )


def _one_to_one(source_begin: int, target_begin: int, count: int, synapse_type: int, weight: float, *, trigger_rule: int = -1):
    return nb.Connection(
        source=[source_begin + i for i in range(count)],
        target=[target_begin + i for i in range(count)],
        synapse_type=synapse_type,
        weight=weight,
        max_weight=weight,
        delay=1,
        trigger_rule=trigger_rule,
    )


def add_connections(network: nb.Network, cfg: CerebellumConfig, index: LayerIndex) -> None:
    rng = random.Random(7)

    def add_gc_pc(gc_begin: int, pc_begin: int) -> None:
        source: list[int] = []
        target: list[int] = []
        weight: list[float] = []
        for pc in range(cfg.n_pc):
            for gc in range(cfg.n_gc):
                source.append(gc_begin + gc)
                target.append(pc_begin + pc)
                weight.append(cfg.w_gcpc_init + rng.random() * cfg.w_gcpc_jitter)
        network.connect(
            nb.Connection(
                source=source,
                target=target,
                weight=weight,
                max_weight=cfg.w_gcpc_max,
                delay=1,
                synapse_rule=0,
            )
        )

    add_gc_pc(index.gc1, index.pc1)
    add_gc_pc(index.gc1, index.pc2)
    add_gc_pc(index.gc2, index.pc3)
    add_gc_pc(index.gc2, index.pc4)

    for source_begin, target_begin in (
        (index.gc1, index.dcn1),
        (index.gc1, index.dcn2),
        (index.gc2, index.dcn3),
        (index.gc2, index.dcn4),
    ):
        network.connect(_dense_projection(source_begin, cfg.n_gc, target_begin, cfg.n_dcn, 0, cfg.w_gc_dcn))

    for source_begin, target_begin in (
        (index.pc1, index.dcn1),
        (index.pc2, index.dcn2),
        (index.pc3, index.dcn3),
        (index.pc4, index.dcn4),
    ):
        network.connect(_one_to_one(source_begin, target_begin, min(cfg.n_pc, cfg.n_dcn), 1, cfg.w_pc_dcn))

    if cfg.n_cf != cfg.n_pc:
        raise ValueError("n_cf must equal n_pc for one-to-one CF->PC trigger routing")
    for source_begin, target_begin in (
        (index.cf1, index.pc1),
        (index.cf2, index.pc2),
        (index.cf3, index.pc3),
        (index.cf4, index.pc4),
    ):
        network.connect(_one_to_one(source_begin, target_begin, cfg.n_cf, 0, 0.0, trigger_rule=0))


def add_learning_rule(network: nb.Network, cfg: CerebellumConfig) -> None:
    network.add_learning_rule(
        nb.LearningRule(
            "CerebullarLearningRule",
            parameters={
                "a1pre": nb.float32(cfg.alpha),
                "a2prepre": nb.float32(cfg.beta),
                "initpos": nb.float32(0.0),
                "maxpos": nb.float32(cfg.control_interval_ms),
                "kernel_step_size": nb.float32(cfg.dt_ms),
                "min_weight": nb.float32(cfg.w_gcpc_min),
                "random_seed": nb.int32(17),
            },
        )
    )


def add_outer_dynamic(network: nb.Network, cfg: CerebellumConfig, index: LayerIndex) -> None:
    gc_joint_0 = [index.gc1 + i for i in range(cfg.n_gc)]
    gc_joint_1 = [index.gc2 + i for i in range(cfg.n_gc)]
    cf_pos_0 = [index.cf1 + i for i in range(cfg.n_cf)]
    cf_neg_0 = [index.cf2 + i for i in range(cfg.n_cf)]
    cf_pos_1 = [index.cf3 + i for i in range(cfg.n_cf)]
    cf_neg_1 = [index.cf4 + i for i in range(cfg.n_cf)]

    network.add_outer_dynamic(
        nb.OuterDynamic.strict_matlab_planar_arm_2dof(
            name="cerebellum_arm",
            update_timestep=cfg.steps_per_control,
            communication_interval=cfg.steps_per_control,
            link_lengths=[cfg.link1, cfg.link2],
            link_masses=[cfg.mass1, cfg.mass2],
            angle_min_deg=[-30.0, 0.1],
            angle_max_deg=[90.0, 150.0],
            velocity_min_deg_s=[-400.0, -400.0],
            velocity_range_deg_s=[800.0, 800.0],
            torque_scale=[cfg.dcn_torque_gain_1, cfg.dcn_torque_gain_2],
            spike_retention_steps=cfg.steps_per_control * 4,
            state_feedback_delay_steps=1,
            state_feedback_repeats_per_update=cfg.gc_repeats_per_control,
            state_feedback_repeat_period_steps=max(1, int(round(4.0 / cfg.dt_ms))),
            error_feedback_delay_steps=0,
            error_feedback_window_steps=cfg.steps_per_control,
            error_feedback_sample_count=cfg.sample_count,
            cf_mix_position=cfg.cf_mix_position,
            spike_cf_max=cfg.spike_cf_max,
            cf_angle_norm_deg=[60.0, 75.0],
            cf_velocity_norm_deg_s=[400.0, 400.0],
            error_feedback_seed=17,
            state_feedback_product=nb.FeedbackProductEncoding(
                bins=[cfg.nn, cfg.nn, cfg.nv, cfg.nv],
                neuron_indices_by_joint=[gc_joint_0, gc_joint_1],
            ),
            cf_positive_neuron_indices_by_joint=[cf_pos_0, cf_pos_1],
            cf_negative_neuron_indices_by_joint=[cf_neg_0, cf_neg_1],
        )
    )

    source: list[int] = []
    target_joint: list[int] = []
    synapse_type: list[int] = []
    weight: list[float] = []
    for i in range(cfg.n_dcn):
        source.extend([index.dcn1 + i, index.dcn2 + i, index.dcn3 + i, index.dcn4 + i])
        target_joint.extend([0, 0, 1, 1])
        synapse_type.extend([0, 1, 0, 1])
        weight.extend([cfg.dcn_torque_gain_1, cfg.dcn_torque_gain_1, cfg.dcn_torque_gain_2, cfg.dcn_torque_gain_2])
    network.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=source,
            target_outer_dynamic=0,
            target_joint=target_joint,
            synapse_type=synapse_type,
            weight=weight,
            delay=0,
        )
    )


def build_network(cfg: CerebellumConfig) -> nb.Network:
    network = nb.Network()
    index = add_layers(network, cfg)
    add_learning_rule(network, cfg)
    add_connections(network, cfg, index)
    add_outer_dynamic(network, cfg, index)
    return network


def hand_position(q1: float, q2: float, cfg: CerebellumConfig) -> tuple[float, float]:
    return (
        cfg.link1 * math.cos(q1) + cfg.link2 * math.cos(q1 + q2),
        cfg.link1 * math.sin(q1) + cfg.link2 * math.sin(q1 + q2),
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_desired_trajectory_analytic(cfg: CerebellumConfig) -> list[StateSample]:
    positions: list[tuple[float, float, float, float]] = []
    total_ms = cfg.control_interval_ms * cfg.sample_count
    for sample in range(cfg.sample_count + 1):
        phase = sample * cfg.control_interval_ms / total_ms
        x = 0.15 * math.cos(2.0 * math.pi * phase) + 0.1
        y = 0.15 * math.sin(2.0 * math.pi * phase) + 0.4
        c2 = clamp((x * x + y * y - cfg.link1 * cfg.link1 - cfg.link2 * cfg.link2) / (2.0 * cfg.link1 * cfg.link2), -1.0, 1.0)
        q2 = math.acos(c2)
        phi = math.acos(
            clamp(
                (cfg.link2 * cfg.link2 - cfg.link1 * cfg.link1 - x * x - y * y) / (-2.0 * cfg.link1 * math.sqrt(x * x + y * y)),
                -1.0,
                1.0,
            )
        )
        q1 = math.atan2(y, x) - phi
        positions.append((q1, q2, x, y))

    dt_s = cfg.control_interval_ms * 0.001
    samples: list[StateSample] = []
    velocities: list[tuple[float, float]] = []
    for index in range(cfg.sample_count):
        velocities.append(((positions[index + 1][0] - positions[index][0]) / dt_s, (positions[index + 1][1] - positions[index][1]) / dt_s))
    velocities.append(velocities[-1])
    for (q1, q2, x, y), (qd1, qd2) in zip(positions, velocities):
        samples.append(StateSample(q1=q1, q2=q2, qd1=qd1, qd2=qd2, x=x, y=y))
    return samples


def write_state_header(handle) -> None:
    handle.write("time_s\tq1\tq2\tqd1\tqd2\tqdes1\tqdes2\tqdd1\tqdd2\ttau1\ttau2\tx\ty\n")


def write_state_row(handle, time_s: float, state: dict, desired: StateSample, cfg: CerebellumConfig) -> None:
    x, y = hand_position(state["q"][0], state["q"][1], cfg)
    values = [
        time_s,
        state["q"][0],
        state["q"][1],
        state["qv"][0],
        state["qv"][1],
        desired.q1,
        desired.q2,
        state["qdd"][0],
        state["qdd"][1],
        state["tau_total"][0],
        state["tau_total"][1],
        x,
        y,
    ]
    handle.write("\t".join(f"{value:.17g}" for value in values) + "\n")


def run_example(cfg: CerebellumConfig, output_dir: Path) -> dict:
    if not nb.backend_info()["native_extension_loaded"]:
        raise RuntimeError("neuronbridge native extension is not built")
    output_dir.mkdir(parents=True, exist_ok=True)
    network = build_network(cfg)
    sim = nb.Simulation(network, nb.SimulationConfig(steps=cfg.total_steps, timestep=cfg.dt_ms))
    sim.init()
    desired = build_desired_trajectory_analytic(cfg)
    initial = desired[1]
    sim.reset_outer_dynamic_state("cerebellum_arm", [initial.q1, initial.q2], [initial.qd1, initial.qd2])
    sim.set_outer_dynamic_desired_state("cerebellum_arm", [initial.q1, initial.q2], [initial.qd1, initial.qd2])
    state_path = output_dir / "joint_state.tsv"
    with state_path.open("w", encoding="utf-8", newline="\n") as handle:
        write_state_header(handle)
        for sample in range(2, cfg.sample_count):
            target = desired[sample]
            sim.set_outer_dynamic_desired_state("cerebellum_arm", [target.q1, target.q2], [target.qd1, target.qd2])
            sim.run(cfg.steps_per_control)
            write_state_row(handle, sample * cfg.control_interval_ms * 0.001, sim.outer_dynamic_state(), target, cfg)

    final_state = sim.outer_dynamic_state()
    summary = {
        "samples": max(0, cfg.sample_count - 2),
        "steps": cfg.total_steps,
        "neurons": network.neuron_count,
        "connections": len(network.connections),
        "outer_dynamics": len(network.outer_dynamics),
        "final_time_step": final_state["time_step"],
        "final_q1": final_state["q"][0],
        "final_q2": final_state["q"][1],
        "final_tau1": final_state["tau_total"][0],
        "final_tau2": final_state["tau_total"][1],
    }
    with (output_dir / "summary.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("metric\tvalue\n")
        for key, value in summary.items():
            handle.write(f"{key}\t{value}\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=artifact_dir("cerebellum_framework"))
    parser.add_argument("--sample-count", type=int, default=CerebellumConfig.sample_count)
    parser.add_argument("--nn", type=int, default=CerebellumConfig.nn)
    parser.add_argument("--nv", type=int, default=CerebellumConfig.nv)
    parser.add_argument("--n-pc", type=int, default=CerebellumConfig.n_pc)
    parser.add_argument("--n-dcn", type=int, default=CerebellumConfig.n_dcn)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CerebellumConfig(sample_count=args.sample_count, nn=args.nn, nv=args.nv, n_pc=args.n_pc, n_cf=args.n_pc, n_dcn=args.n_dcn)
    summary = run_example(cfg, args.output_dir)
    for key, value in summary.items():
        print(f"{key}={value}")
    print(f"output_dir={args.output_dir}")
    print("cerebellum framework Python migration finished.")


if __name__ == "__main__":
    main()
