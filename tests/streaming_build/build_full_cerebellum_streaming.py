"""Build and smoke-test the full 2-DOF cerebellum through ``.nbnet`` streaming.

The default dimensions match ``cerebellum_framework_arm_repro.cpp``.  The
connection table is generated in bounded batches; no complete Python Network or
connection list is created.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import neuronbridge as nb


CONDUCTANCE_SCALE = 1.0 / 6.0


@dataclass(frozen=True)
class FullCerebellumConfig:
    dt_ms: float = 0.1
    control_interval_ms: float = 20.0
    sample_count: int = 50
    nn: int = 15
    nv: int = 10
    n_pc: int = 200
    n_cf: int = 200
    n_dcn: int = 200
    gc_repeats_per_control: int = 5
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
        return self.sample_count * self.steps_per_control


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


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory() -> dict[str, int]:
    if not hasattr(ctypes, "windll"):
        return {}
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return {}
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
        "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
    }


def lif_layer(count: int, *, output: bool = False) -> nb.NeuronLayer:
    # ``output`` is layer metadata, not a neuron-model parameter.  Passing it
    # through ``NeuronLayer.lif_double(..., output=True)`` silently places it
    # in the model parameter map and leaves Neuron::IsOutput false.
    return nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double",
        count,
        output=output,
        parameters={
            "V_rest": nb.float32(-70.0),
            "V_reset": nb.float32(-80.0),
            "V_th": nb.float32(-40.0),
            "tau": nb.float32(10.0 / 6.0),
            "R": nb.float32(1.0),
            "t_ref": nb.int32(10),
            "gexc_tau": nb.float32(1.0),
            "ginh_tau": nb.float32(1.0),
            "Eexc": nb.float32(0.0),
            "Einh": nb.float32(-80.0),
            "random_mu": nb.float32_array4([-10.0, 0.0, 0.0, 0.0]),
            "random_sigma": nb.float32_array4([0.0, 0.0, 0.0, 0.0]),
        },
    )


def add_metadata(
    builder: nb.NbnetDescriptionBuilder,
    cfg: FullCerebellumConfig,
    *,
    include_outer_dynamic: bool = True,
) -> LayerIndex:
    starts: list[int] = []
    next_id = 0
    for count in [cfg.n_gc, cfg.n_gc, cfg.n_cf, cfg.n_cf, cfg.n_cf, cfg.n_cf]:
        starts.append(next_id)
        next_id += count
        builder.add_layer(nb.NeuronLayer.input_spike(count))
    for _ in range(4):
        starts.append(next_id)
        next_id += cfg.n_pc
        builder.add_layer(lif_layer(cfg.n_pc))
    for _ in range(4):
        starts.append(next_id)
        next_id += cfg.n_dcn
        builder.add_layer(lif_layer(cfg.n_dcn, output=True))
    index = LayerIndex(*starts)

    builder.add_learning_rule(
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

    if not include_outer_dynamic:
        return index

    builder.add_outer_dynamic(
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
                neuron_indices_by_joint=[
                    list(range(index.gc1, index.gc1 + cfg.n_gc)),
                    list(range(index.gc2, index.gc2 + cfg.n_gc)),
                ],
            ),
            cf_positive_neuron_indices_by_joint=[
                list(range(index.cf1, index.cf1 + cfg.n_cf)),
                list(range(index.cf3, index.cf3 + cfg.n_cf)),
            ],
            cf_negative_neuron_indices_by_joint=[
                list(range(index.cf2, index.cf2 + cfg.n_cf)),
                list(range(index.cf4, index.cf4 + cfg.n_cf)),
            ],
        )
    )
    source: list[int] = []
    target_joint: list[int] = []
    synapse_type: list[int] = []
    weight: list[float] = []
    for neuron in range(cfg.n_dcn):
        source.extend([index.dcn1 + neuron, index.dcn2 + neuron, index.dcn3 + neuron, index.dcn4 + neuron])
        target_joint.extend([0, 0, 1, 1])
        synapse_type.extend([0, 1, 0, 1])
        weight.extend(
            [cfg.dcn_torque_gain_1, cfg.dcn_torque_gain_1, cfg.dcn_torque_gain_2, cfg.dcn_torque_gain_2]
        )
    builder.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=source,
            target_outer_dynamic=0,
            target_joint=target_joint,
            synapse_type=synapse_type,
            weight=weight,
            delay=0,
        )
    )
    return index


def append_projection(
    builder: nb.NbnetDescriptionBuilder,
    *,
    source_begin: int,
    source_count: int,
    target_begin: int,
    target_count: int,
    batch_size: int,
    weight: float,
    max_weight: float,
    synapse_type: int = 0,
    synapse_rule: int = -1,
    trigger_rule: int = -1,
    rng: random.Random | None = None,
    jitter: float = 0.0,
) -> int:
    total = source_count * target_count
    for first in range(0, total, batch_size):
        count = min(batch_size, total - first)
        source = [source_begin + ((first + offset) % source_count) for offset in range(count)]
        target = [target_begin + ((first + offset) // source_count) for offset in range(count)]
        weights: float | list[float]
        if rng is None:
            weights = weight
        else:
            weights = [weight + rng.random() * jitter for _ in range(count)]
        builder.append_connections(
            source,
            target,
            synapse_type=synapse_type,
            weight=weights,
            max_weight=max_weight,
            delay=1,
            synapse_rule=synapse_rule,
            trigger_rule=trigger_rule,
        )
    return total


def append_one_to_one(
    builder: nb.NbnetDescriptionBuilder,
    source_begin: int,
    target_begin: int,
    count: int,
    *,
    synapse_type: int,
    weight: float,
    trigger_rule: int = -1,
) -> int:
    builder.append_connections(
        list(range(source_begin, source_begin + count)),
        list(range(target_begin, target_begin + count)),
        synapse_type=synapse_type,
        weight=weight,
        max_weight=weight,
        delay=1,
        trigger_rule=trigger_rule,
    )
    return count


def generate_nbnet(
    path: Path,
    cfg: FullCerebellumConfig,
    batch_size: int,
    *,
    include_outer_dynamic: bool = True,
) -> nb.NbnetBuildResult:
    rng = random.Random(7)
    with nb.NbnetDescriptionBuilder(path, options=nb.NbnetWriteOptions(overwrite=True)) as builder:
        index = add_metadata(builder, cfg, include_outer_dynamic=include_outer_dynamic)
        written = 0
        for gc_begin, pc_begin in (
            (index.gc1, index.pc1),
            (index.gc1, index.pc2),
            (index.gc2, index.pc3),
            (index.gc2, index.pc4),
        ):
            written += append_projection(
                builder,
                source_begin=gc_begin,
                source_count=cfg.n_gc,
                target_begin=pc_begin,
                target_count=cfg.n_pc,
                batch_size=batch_size,
                weight=cfg.w_gcpc_init,
                max_weight=cfg.w_gcpc_max,
                synapse_rule=0,
                rng=rng,
                jitter=cfg.w_gcpc_jitter,
            )
            print(f"generated_connections={written}", flush=True)
        for gc_begin, dcn_begin in (
            (index.gc1, index.dcn1),
            (index.gc1, index.dcn2),
            (index.gc2, index.dcn3),
            (index.gc2, index.dcn4),
        ):
            written += append_projection(
                builder,
                source_begin=gc_begin,
                source_count=cfg.n_gc,
                target_begin=dcn_begin,
                target_count=cfg.n_dcn,
                batch_size=batch_size,
                weight=cfg.w_gc_dcn,
                max_weight=cfg.w_gc_dcn,
            )
            print(f"generated_connections={written}", flush=True)
        for pc_begin, dcn_begin in (
            (index.pc1, index.dcn1),
            (index.pc2, index.dcn2),
            (index.pc3, index.dcn3),
            (index.pc4, index.dcn4),
        ):
            written += append_one_to_one(
                builder,
                pc_begin,
                dcn_begin,
                min(cfg.n_pc, cfg.n_dcn),
                synapse_type=1,
                weight=cfg.w_pc_dcn,
            )
        for cf_begin, pc_begin in (
            (index.cf1, index.pc1),
            (index.cf2, index.pc2),
            (index.cf3, index.pc3),
            (index.cf4, index.pc4),
        ):
            written += append_one_to_one(
                builder,
                cf_begin,
                pc_begin,
                cfg.n_cf,
                synapse_type=0,
                weight=0.0,
                trigger_rule=0,
            )
        result = builder.finalize()
    if written != result.connection_count:
        raise RuntimeError(f"connection count mismatch: generated {written}, file has {result.connection_count}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/full_cerebellum_streaming"))
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--memory-budget-mb", type=int, default=64)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument(
        "--external-arm-server",
        action="store_true",
        help="omit OuterDynamic metadata so an external communication server owns the arm",
    )
    parser.add_argument("--nn", type=int, default=15)
    parser.add_argument("--nv", type=int, default=10)
    parser.add_argument("--n-pc", type=int, default=200)
    parser.add_argument("--n-dcn", type=int, default=200)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.memory_budget_mb <= 0:
        parser.error("batch size and memory budget must be positive")

    cfg = FullCerebellumConfig(
        nn=args.nn,
        nv=args.nv,
        n_pc=args.n_pc,
        n_cf=args.n_pc,
        n_dcn=args.n_dcn,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nbnet_path = args.output_dir / "full_cerebellum.nbnet"
    report_path = args.output_dir / "build_report.json"
    report: dict[str, object] = {
        "config": asdict(cfg),
        "n_gc_per_joint": cfg.n_gc,
        "expected_neurons": 2 * cfg.n_gc + 4 * cfg.n_cf + 4 * cfg.n_pc + 4 * cfg.n_dcn,
        "expected_connections": (
            4 * cfg.n_gc * (cfg.n_pc + cfg.n_dcn)
            + 4 * min(cfg.n_pc, cfg.n_dcn)
            + 4 * cfg.n_cf
        ),
        "batch_size": args.batch_size,
        "memory_budget_mb": args.memory_budget_mb,
        "include_outer_dynamic": not args.external_arm_server,
        "backend": nb.backend_info(),
        "started_unix": time.time(),
    }

    if not args.skip_generate:
        started = time.perf_counter()
        result = generate_nbnet(
            nbnet_path,
            cfg,
            args.batch_size,
            include_outer_dynamic=not args.external_arm_server,
        )
        report["generation_seconds"] = time.perf_counter() - started
        report["nbnet"] = {
            "path": str(result.path),
            "file_bytes": result.file_bytes,
            "neuron_count": result.neuron_count,
            "connection_count": result.connection_count,
            "sha256": result.sha256,
        }
        report["memory_after_generation"] = process_memory()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    info = nb.nbnet.inspect(nbnet_path, verify_checksum=False)
    report["file_info"] = {
        "path": str(info.path),
        "file_bytes": info.file_bytes,
        "metadata_bytes": info.metadata_bytes,
        "neuron_count": info.neuron_count,
        "connection_count": info.connection_count,
    }
    if info.neuron_count != report["expected_neurons"] or info.connection_count != report["expected_connections"]:
        raise RuntimeError("generated file does not match the full cerebellum dimensions")

    if args.external_arm_server and not args.skip_run:
        raise RuntimeError("--external-arm-server requires --skip-run; the network has no internal plant")

    started = time.perf_counter()
    sim = nb.Simulation(
        nbnet_path,
        nb.SimulationConfig(steps=cfg.total_steps, timestep=cfg.dt_ms),
        build_options=nb.StreamingBuildOptions(
            memory_budget_mb=args.memory_budget_mb,
            mmap=True,
            verify_checksum=True,
        ),
    )
    report["simulation_construct_seconds"] = time.perf_counter() - started
    report["build_stats"] = sim.build_stats
    report["memory_after_construct"] = process_memory()
    if sim.build_stats.get("runtime_build_path") != "streaming_direct":
        raise RuntimeError(f"unexpected build path: {sim.build_stats}")

    started = time.perf_counter()
    sim.init()
    report["simulation_init_seconds"] = time.perf_counter() - started
    report["memory_after_init"] = process_memory()
    if not args.skip_run and args.smoke_steps > 0:
        started = time.perf_counter()
        sim.run(args.smoke_steps)
        report["smoke_steps"] = args.smoke_steps
        report["smoke_run_seconds"] = time.perf_counter() - started
        report["outer_dynamic_state"] = sim.outer_dynamic_state()
        report["memory_after_smoke"] = process_memory()

    report["completed_unix"] = time.time()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
