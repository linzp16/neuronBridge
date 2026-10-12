"""Train a six-axis ROKAE xMate arm with the cerebellar learning topology.

This is the 6-DOF counterpart of ``cerebellum_framework.py``.  Every joint has
one GC state population and two signed PC/CF/DCN channels.  The ROKAE
OuterDynamic emits GC and CF feedback internally, while the positive/negative
DCN populations are decoded as joint torque. Direct PD torque is off by default;
the optional --pd-assist flag explicitly opts into the historical hybrid mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import neuronbridge as nb


JOINT_COUNT = 6
CONDUCTANCE_SCALE = 1.0 / 6.0


@dataclass(frozen=True)
class Config:
    epochs: int = 20
    samples_per_epoch: int = 100
    dt_ms: float = 0.1
    control_interval_ms: float = 20.0
    nn: int = 3
    nv: int = 2
    n_pc: int = 48
    n_cf: int = 48
    n_dcn: int = 48
    gc_repeats_per_control: int = 5
    w_gc_dcn: float = 12.0 * CONDUCTANCE_SCALE
    w_pc_dcn: float = 6.0
    w_gcpc_init: float = 2.2
    w_gcpc_jitter: float = 0.1 * CONDUCTANCE_SCALE
    w_gcpc_max: float = 2.5
    w_gcpc_min: float = 1.5
    alpha: float = 6.0e-3 * CONDUCTANCE_SCALE
    beta: float = -0.5 * CONDUCTANCE_SCALE
    torque_gain: tuple[float, ...] = (0.025, 0.10, 0.0625, 0.03, 0.02, 0.00625)
    cf_mix_position: float = 0.8
    spike_cf_max: float = 15.0
    pd_assist: bool = False
    pd_kp: tuple[float, ...] = (45., 45., 35., 20., 12., 8.)
    pd_kd: tuple[float, ...] = (10., 10., 8., 5., 3., 2.)
    cf_angle_norm_deg: float = 30.
    cf_velocity_norm_deg_s: float = 180.
    feedback_sample_count: int | None = None
    angle_min_deg: tuple[float, ...] | None = None
    angle_max_deg: tuple[float, ...] | None = None
    velocity_min_deg_s: tuple[float, ...] | None = None
    velocity_range_deg_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if min(self.epochs, self.samples_per_epoch, self.nn, self.nv,
               self.n_pc, self.n_cf, self.n_dcn, self.gc_repeats_per_control) <= 0:
            raise ValueError("Epochs, samples, bins and population sizes must be positive")
        if self.n_pc != self.n_cf or self.n_pc != self.n_dcn:
            raise ValueError("Use equal PC/CF/DCN sizes for complete one-to-one routing")
        if len(self.torque_gain) != JOINT_COUNT or any(not math.isfinite(g) or g < 0 for g in self.torque_gain):
            raise ValueError("torque_gain must contain six finite non-negative gains")
        if not self.w_gcpc_min <= self.w_gcpc_init <= self.w_gcpc_init + self.w_gcpc_jitter <= self.w_gcpc_max:
            raise ValueError("Initial plastic weights must lie within their bounds")
        for values in (self.pd_kp, self.pd_kd):
            if len(values) != JOINT_COUNT or any(not math.isfinite(x) or x < 0 for x in values):
                raise ValueError('PD gains must contain six finite non-negative values')
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.cf_angle_norm_deg, self.cf_velocity_norm_deg_s, self.dt_ms, self.control_interval_ms)):
            raise ValueError('Time intervals and CF normalization must be finite and positive')
        if not math.isfinite(self.alpha) or not math.isfinite(self.beta) or self.alpha < 0 or self.beta > 0:
            raise ValueError('Learning requires non-negative LTP and non-positive LTD')
        if self.feedback_sample_count is not None and self.feedback_sample_count <= 0:
            raise ValueError('Feedback sample count must be positive')
        if not 0 <= self.cf_mix_position <= 1:
            raise ValueError('CF position mixture must lie in [0, 1]')
        for key in ('angle_min_deg', 'angle_max_deg', 'velocity_min_deg_s', 'velocity_range_deg_s'):
            values = getattr(self, key)
            if values is not None and (len(values) != JOINT_COUNT or not all(math.isfinite(x) for x in values)):
                raise ValueError(f'{key} must contain six finite values')
        if (self.angle_min_deg is None) != (self.angle_max_deg is None):
            raise ValueError('Provide both angle encoding bounds')
        if self.angle_min_deg is not None and any(a >= b for a,b in zip(self.angle_min_deg,self.angle_max_deg)):
            raise ValueError('Angle encoding bounds must increase')
        if self.velocity_range_deg_s is not None and any(x <= 0 for x in self.velocity_range_deg_s):
            raise ValueError('Velocity encoding ranges must be positive')

    @property
    def n_gc(self) -> int:
        return self.nn * self.nn * self.nv * self.nv

    @property
    def steps_per_control(self) -> int:
        return int(round(self.control_interval_ms / self.dt_ms))


@dataclass(frozen=True)
class JointLayers:
    gc: int
    cf_positive: int
    cf_negative: int
    pc_positive: int
    pc_negative: int
    dcn_positive: int
    dcn_negative: int


def lif_layer(count: int, *, output: bool = False) -> nb.NeuronLayer:
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


def add_layers(network: nb.Network, cfg: Config) -> list[JointLayers]:
    next_id = 0
    layers: list[JointLayers] = []
    for _joint in range(JOINT_COUNT):
        starts: list[int] = []
        for count, kind in (
            (cfg.n_gc, "input"),
            (cfg.n_cf, "input"),
            (cfg.n_cf, "input"),
            (cfg.n_pc, "lif"),
            (cfg.n_pc, "lif"),
            (cfg.n_dcn, "output"),
            (cfg.n_dcn, "output"),
        ):
            starts.append(next_id)
            next_id += count
            if kind == "input":
                network.add_layer(nb.NeuronLayer.input_spike(count))
            else:
                network.add_layer(lif_layer(count, output=kind == "output"))
        layers.append(JointLayers(*starts))
    return layers


def dense_projection(source_begin: int, source_count: int, target_begin: int, target_count: int,
                     synapse_type: int, weight: float) -> nb.Connection:
    return nb.Connection(
        source=[source_begin + source for target in range(target_count) for source in range(source_count)],
        target=[target_begin + target for target in range(target_count) for source in range(source_count)],
        synapse_type=synapse_type,
        weight=weight,
        max_weight=weight,
        delay=1,
    )


def one_to_one(source_begin: int, target_begin: int, count: int, synapse_type: int,
               weight: float, *, trigger_rule: int = -1) -> nb.Connection:
    return nb.Connection(
        source=[source_begin + index for index in range(count)],
        target=[target_begin + index for index in range(count)],
        synapse_type=synapse_type,
        weight=weight,
        max_weight=weight,
        delay=1,
        trigger_rule=trigger_rule,
    )


def add_joint_connections(network: nb.Network, cfg: Config, layer: JointLayers, rng: random.Random) -> None:
    for pc_begin in (layer.pc_positive, layer.pc_negative):
        source: list[int] = []
        target: list[int] = []
        weights: list[float] = []
        for pc in range(cfg.n_pc):
            for gc in range(cfg.n_gc):
                source.append(layer.gc + gc)
                target.append(pc_begin + pc)
                weights.append(cfg.w_gcpc_init + rng.random() * cfg.w_gcpc_jitter)
        network.connect(nb.Connection(
            source=source,
            target=target,
            weight=weights,
            max_weight=cfg.w_gcpc_max,
            delay=1,
            synapse_rule=0,
        ))

    for dcn_begin in (layer.dcn_positive, layer.dcn_negative):
        network.connect(dense_projection(layer.gc, cfg.n_gc, dcn_begin, cfg.n_dcn, 0, cfg.w_gc_dcn))
    network.connect(one_to_one(layer.pc_positive, layer.dcn_positive,
                               min(cfg.n_pc, cfg.n_dcn), 1, cfg.w_pc_dcn))
    network.connect(one_to_one(layer.pc_negative, layer.dcn_negative,
                               min(cfg.n_pc, cfg.n_dcn), 1, cfg.w_pc_dcn))
    network.connect(one_to_one(layer.cf_positive, layer.pc_positive, cfg.n_cf, 0, 0.0, trigger_rule=0))
    network.connect(one_to_one(layer.cf_negative, layer.pc_negative, cfg.n_cf, 0, 0.0, trigger_rule=0))


def build_network(cfg: Config, urdf_path: Path) -> tuple[nb.Network, list[JointLayers], list[int]]:
    if cfg.n_cf != cfg.n_pc:
        raise ValueError("n_cf must equal n_pc for one-to-one CF->PC trigger routing")
    network = nb.Network()
    layers = add_layers(network, cfg)
    network.add_learning_rule(nb.LearningRule(
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
    ))
    rng = random.Random(7)
    for layer in layers:
        add_joint_connections(network, cfg, layer, rng)

    plastic_per_channel = cfg.n_gc * cfg.n_pc
    connections_per_joint = (
        2 * plastic_per_channel
        + 2 * cfg.n_gc * cfg.n_dcn
        + 2 * min(cfg.n_pc, cfg.n_dcn)
        + 2 * cfg.n_cf
    )
    plastic_indices = [
        joint * connections_per_joint + local
        for joint in range(JOINT_COUNT)
        for local in range(2 * plastic_per_channel)
    ]

    gc_groups = [[layer.gc + index for index in range(cfg.n_gc)] for layer in layers]
    cf_positive = [[layer.cf_positive + index for index in range(cfg.n_cf)] for layer in layers]
    cf_negative = [[layer.cf_negative + index for index in range(cfg.n_cf)] for layer in layers]
    encoding_ranges = {key: list(getattr(cfg, key)) for key in (
        'angle_min_deg', 'angle_max_deg', 'velocity_min_deg_s', 'velocity_range_deg_s')
        if getattr(cfg, key) is not None}
    network.add_outer_dynamic(nb.OuterDynamic.rokae_arm(
        urdf_path=str(urdf_path),
        name="rokae_cerebellum",
        trajectory_mode="hold",
        enable_pd_control=cfg.pd_assist,
        update_timestep=cfg.steps_per_control,
        communication_interval=cfg.steps_per_control,
        pd_kp=list(cfg.pd_kp),
        pd_kd=list(cfg.pd_kd),
        joint_damping=[0.15] * JOINT_COUNT,
        spike_retention_steps=cfg.steps_per_control * 4,
        state_feedback_delay_steps=1,
        state_feedback_repeats_per_update=cfg.gc_repeats_per_control,
        state_feedback_repeat_period_steps=max(1, int(round(4.0 / cfg.dt_ms))),
        error_feedback_delay_steps=0,
        error_feedback_window_steps=cfg.steps_per_control,
        error_feedback_sample_count=cfg.feedback_sample_count or cfg.samples_per_epoch,
        cf_mix_position=cfg.cf_mix_position,
        spike_cf_max=cfg.spike_cf_max,
        cf_angle_norm_deg=[cfg.cf_angle_norm_deg] * JOINT_COUNT,
        cf_velocity_norm_deg_s=[cfg.cf_velocity_norm_deg_s] * JOINT_COUNT,
        **encoding_ranges,
        error_feedback_seed=17,
        state_feedback_product=nb.FeedbackProductEncoding(
            bins=[cfg.nn, cfg.nn, cfg.nv, cfg.nv],
            neuron_indices_by_joint=gc_groups,
        ),
        cf_positive_neuron_indices_by_joint=cf_positive,
        cf_negative_neuron_indices_by_joint=cf_negative,
    ))

    source: list[int] = []
    target_joint: list[int] = []
    synapse_type: list[int] = []
    weights: list[float] = []
    for joint, layer in enumerate(layers):
        for index in range(cfg.n_dcn):
            source.extend([layer.dcn_positive + index, layer.dcn_negative + index])
            target_joint.extend([joint, joint])
            synapse_type.extend([0, 1])
            weights.extend([cfg.torque_gain[joint], cfg.torque_gain[joint]])
    network.connect_outer_dynamic(nb.OuterDynamicConnection(
        source=source,
        target_outer_dynamic=0,
        target_joint=target_joint,
        synapse_type=synapse_type,
        weight=weights,
        delay=0,
    ))
    return network, layers, plastic_indices


def desired_trajectory(cfg: Config) -> list[tuple[list[float], list[float]]]:
    center_deg = [0.0, 20.0, -70.0, 0.0, 45.0, 0.0]
    amplitude_deg = [12.0, 10.0, 12.0, 15.0, 10.0, 15.0]
    phase = [0.0, 0.5, 1.0, 1.5, 0.75, 1.25]
    duration_s = cfg.samples_per_epoch * cfg.control_interval_ms * 0.001
    omega = 2.0 * math.pi / duration_s
    trajectory: list[tuple[list[float], list[float]]] = []
    for sample in range(cfg.samples_per_epoch):
        time_s = sample * cfg.control_interval_ms * 0.001
        q = [math.radians(center_deg[joint] + amplitude_deg[joint] * math.sin(omega * time_s + phase[joint]))
             for joint in range(JOINT_COUNT)]
        qd = [math.radians(amplitude_deg[joint]) * omega * math.cos(omega * time_s + phase[joint])
              for joint in range(JOINT_COUNT)]
        trajectory.append((q, qd))
    return trajectory


def train(cfg: Config, urdf_path: Path, output_dir: Path, *, debug_monitor: bool = False,
          trajectory=None, initial_weights: Path | None = None) -> list[dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    network, _layers, plastic_indices = build_network(cfg, urdf_path)
    sim = nb.Simulation(
        network,
        nb.SimulationConfig(
            steps=cfg.steps_per_control * cfg.samples_per_epoch,
            timestep=cfg.dt_ms,
        ),
    )
    if debug_monitor:
        sim.enable_debug_monitor(nb.DebugMonitorConfig(
            output_dir=output_dir / "debug_monitor",
            sample_interval_steps=1,
            flush_interval_steps=cfg.steps_per_control,
            record_spikes=True,
            record_state=False,
            record_weights=False,
            record_pending_channels=False,
            record_outer_dynamic_state=True,
            all_neurons=True,
        ))
    sim.init()
    if initial_weights is not None:
        sim.load_weights(initial_weights)
    initial_plastic_weights = [sim.get_connection_weight(index) for index in plastic_indices]
    trajectory = desired_trajectory(cfg) if trajectory is None else trajectory
    if len(trajectory) != cfg.samples_per_epoch:
        raise ValueError('Trajectory length must match samples_per_epoch')
    metrics: list[dict[str, float]] = []
    traces_path = output_dir / "joint_tracking.csv"
    with traces_path.open("w", newline="", encoding="utf-8") as trace_handle:
        writer = csv.writer(trace_handle)
        writer.writerow(["epoch", "sample", "joint", "q_des_rad", "q_rad", "error_rad",
                         "qd_des_rad_s", "qd_rad_s", "tau_total_nm"])
        for epoch in range(cfg.epochs):
            if epoch:
                sim.reset()
            initial_q, initial_qd = trajectory[0]
            sim.reset_outer_dynamic_state("rokae_cerebellum", initial_q, initial_qd)
            squared_error = [0.0] * JOINT_COUNT
            for sample, (q_des, qd_des) in enumerate(trajectory):
                sim.set_outer_dynamic_desired_state("rokae_cerebellum", q_des, qd_des)
                sim.run(cfg.steps_per_control)
                state = sim.outer_dynamic_state()
                for joint in range(JOINT_COUNT):
                    error = q_des[joint] - state["q"][joint]
                    squared_error[joint] += error * error
                    writer.writerow([epoch + 1, sample, joint + 1, q_des[joint], state["q"][joint], error,
                                     qd_des[joint], state['qv'][joint], state['tau_total'][joint]])
            joint_rmse = [math.sqrt(value / cfg.samples_per_epoch) for value in squared_error]
            row = {"epoch": float(epoch + 1), "mean_rmse_rad": sum(joint_rmse) / JOINT_COUNT}
            row.update({f"joint_{joint + 1}_rmse_rad": joint_rmse[joint] for joint in range(JOINT_COUNT)})
            metrics.append(row)
            print(f"epoch={epoch + 1} mean_rmse_rad={row['mean_rmse_rad']:.6f}", flush=True)

    metrics_path = output_dir / "epoch_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    sim.save_weights(output_dir / "rokae_cerebellum_weights.txt")
    final_plastic_weights = [sim.get_connection_weight(index) for index in plastic_indices]
    weight_delta = [final - initial for initial, final in zip(initial_plastic_weights, final_plastic_weights)]
    changed_count = sum(abs(delta) > 1.0e-7 for delta in weight_delta)
    with (output_dir / "training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["plastic_connection_count", len(plastic_indices)])
        writer.writerow(["changed_plastic_connection_count", changed_count])
        writer.writerow(["mean_absolute_weight_change", sum(abs(delta) for delta in weight_delta) / len(weight_delta)])
        writer.writerow(["max_absolute_weight_change", max(abs(delta) for delta in weight_delta)])
        writer.writerow(["initial_mean_rmse_rad", metrics[0]["mean_rmse_rad"]])
        writer.writerow(["final_mean_rmse_rad", metrics[-1]["mean_rmse_rad"]])
    print(f"changed_plastic_connections={changed_count}/{len(plastic_indices)}")
    print(f"max_absolute_weight_change={max(abs(delta) for delta in weight_delta):.6f}")
    plot_metrics(metrics, output_dir / "training_rmse.png")
    return metrics


def plot_metrics(metrics: list[dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [int(row["epoch"]) for row in metrics]
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for joint in range(JOINT_COUNT):
        axis.plot(epochs, [math.degrees(row[f"joint_{joint + 1}_rmse_rad"]) for row in metrics],
                  linewidth=1.4, alpha=0.72, label=f"Joint {joint + 1}")
    axis.plot(epochs, [math.degrees(row["mean_rmse_rad"]) for row in metrics],
              color="black", linewidth=2.5, label="Mean")
    axis.set(xlabel="Epoch", ylabel="RMSE (deg)", title="ROKAE 6-DOF cerebellar training")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def default_urdf() -> Path:
    configured = os.environ.get("NEURONBRIDGE_ROKAE_URDF")
    if configured:
        return Path(configured)
    return Path(r"D:\Pinocchio\Model\xMateSR3C_description\urdf\xMateSR3C.urdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=default_urdf())
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--samples-per-epoch", type=int, default=Config.samples_per_epoch)
    parser.add_argument("--nn", type=int, default=Config.nn, help="position bins per desired/actual angle")
    parser.add_argument("--nv", type=int, default=Config.nv, help="velocity bins per desired/actual velocity")
    parser.add_argument("--n-pc", type=int, default=Config.n_pc)
    parser.add_argument("--n-dcn", type=int, default=Config.n_dcn)
    parser.add_argument("--gc-repeats", type=int, default=Config.gc_repeats_per_control)
    parser.add_argument("--w-gc-dcn", type=float, default=Config.w_gc_dcn)
    parser.add_argument("--w-pc-dcn", type=float, default=Config.w_pc_dcn)
    parser.add_argument("--w-gcpc-init", type=float, default=Config.w_gcpc_init)
    parser.add_argument("--w-gcpc-jitter", type=float, default=Config.w_gcpc_jitter)
    parser.add_argument("--w-gcpc-min", type=float, default=Config.w_gcpc_min)
    parser.add_argument("--w-gcpc-max", type=float, default=Config.w_gcpc_max)
    parser.add_argument(
        "--torque-gain",
        type=float,
        nargs=JOINT_COUNT,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=Config.torque_gain,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rokae_cerebellum_6dof"))
    pd_group = parser.add_mutually_exclusive_group()
    pd_group.add_argument("--pd-assist", action="store_true", help="Explicitly enable direct PD torque (off by default)")
    pd_group.add_argument("--no-pd-assist", action="store_true", help="Compatibility alias; PD is off by default")
    parser.add_argument("--debug-monitor", action="store_true")
    parser.add_argument("--disable-learning", action="store_true")
    args = parser.parse_args()
    if not args.urdf.is_file():
        raise FileNotFoundError(f"ROKAE URDF does not exist: {args.urdf}")
    cfg = Config(
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        nn=args.nn,
        nv=args.nv,
        n_pc=args.n_pc,
        n_cf=args.n_pc,
        n_dcn=args.n_dcn,
        gc_repeats_per_control=args.gc_repeats,
        w_gc_dcn=args.w_gc_dcn,
        w_pc_dcn=args.w_pc_dcn,
        w_gcpc_init=args.w_gcpc_init,
        w_gcpc_jitter=args.w_gcpc_jitter,
        w_gcpc_min=args.w_gcpc_min,
        w_gcpc_max=args.w_gcpc_max,
        torque_gain=tuple(args.torque_gain),
        alpha=0.0 if args.disable_learning else Config.alpha,
        beta=0.0 if args.disable_learning else Config.beta,
        pd_assist=args.pd_assist,
    )
    metrics = train(
        cfg,
        args.urdf.resolve(),
        args.output_dir.resolve(),
        debug_monitor=args.debug_monitor,
    )
    print(f"initial_mean_rmse_rad={metrics[0]['mean_rmse_rad']:.6f}")
    print(f"final_mean_rmse_rad={metrics[-1]['mean_rmse_rad']:.6f}")
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
