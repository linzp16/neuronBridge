"""Handwriting stage-2 helpers mirrored from the C++ example."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
import math

from _handwriting_stage1 import ArmParameters
from _handwriting_stage1 import StrokePath
from _handwriting_stage1 import TorqueTrajectory
from _handwriting_stage1 import SimulatedTrajectory
from _handwriting_stage1 import build_desired_joint_trajectory
from _handwriting_stage1 import mean_squared_error
from _handwriting_stage1 import replay_forward_dynamics


@dataclass(frozen=True)
class Stage2Config:
    n_ext: int = 100
    n_cm: int = 400
    n_pop: int = 8
    total_time_ms: int = 4000
    dt_ms: float = 0.1
    refractory_ms: int = 2
    g_ext_cm: float = 10.0
    g_l: float = 25.0
    cm_membrane_capacitance: float = 500.0
    e_ampa: float = 0.0
    e_leak: float = -70.0
    v_th: float = -50.0
    v_reset: float = -60.0
    v_spike: float = 20.0
    tau_ampa: float = 2.0


@dataclass(frozen=True)
class Shape2D:
    rows: int
    cols: int


@dataclass(frozen=True)
class Shape3D:
    dim0: int
    dim1: int
    dim2: int


@dataclass(frozen=True)
class BinarySpikeSeries:
    shape: Shape2D
    values: bytes


@dataclass(frozen=True)
class BinaryWeightWindows:
    shape: Shape3D
    values: array


@dataclass(frozen=True)
class CmForwardResult:
    population_counts: list[int]
    simulated_windows: int


def load_shape_2d(path: Path) -> Shape2D:
    dims = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if len(dims) != 2:
        raise ValueError(f"unexpected rank in shape file {path}")
    return Shape2D(rows=dims[0], cols=dims[1])


def load_shape_3d(path: Path) -> Shape3D:
    dims = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if len(dims) != 3:
        raise ValueError(f"unexpected rank in shape file {path}")
    return Shape3D(dim0=dims[0], dim1=dims[1], dim2=dims[2])


def load_binary_spike_series(data_path: Path, shape_path: Path) -> BinarySpikeSeries:
    shape = load_shape_2d(shape_path)
    values = data_path.read_bytes()
    expected = shape.rows * shape.cols
    if len(values) != expected:
        raise ValueError(f"{data_path} has {len(values)} bytes; expected {expected}")
    return BinarySpikeSeries(shape=shape, values=values)


def load_binary_weight_windows(data_path: Path, shape_path: Path, *, max_windows: int | None = None) -> BinaryWeightWindows:
    shape = load_shape_3d(shape_path)
    window_count = shape.dim1 if max_windows is None else min(shape.dim1, int(max_windows))
    expected = shape.dim0 * window_count * shape.dim2
    values = array("f")
    with data_path.open("rb") as handle:
        values.fromfile(handle, expected)
    if len(values) != expected:
        raise ValueError(f"{data_path} has {len(values)} floats; expected {expected}")
    return BinaryWeightWindows(shape=Shape3D(shape.dim0, window_count, shape.dim2), values=values)


def find_existing_stage2_input(directory: Path, stem: str, extension: str) -> Path:
    return directory / f"{stem}{extension}"


def build_external_spike_times(spikes: BinarySpikeSeries, *, max_steps: int | None = None) -> list[int]:
    rows = spikes.shape.rows
    cols = spikes.shape.cols if max_steps is None else min(spikes.shape.cols, int(max_steps))
    times: list[int] = []
    for col in range(cols):
        base = col * rows
        for row in range(rows):
            if spikes.values[base + row] != 0:
                times.append(col + 1)
    return times


def build_external_spike_neurons(spikes: BinarySpikeSeries, *, max_steps: int | None = None) -> list[int]:
    rows = spikes.shape.rows
    cols = spikes.shape.cols if max_steps is None else min(spikes.shape.cols, int(max_steps))
    neurons: list[int] = []
    for col in range(cols):
        base = col * rows
        for row in range(rows):
            if spikes.values[base + row] != 0:
                neurons.append(row)
    return neurons


def population_directions(n_pop: int) -> list[tuple[float, float]]:
    return [
        (math.cos((360.0 * index / n_pop) * math.pi / 180.0), math.sin((360.0 * index / n_pop) * math.pi / 180.0))
        for index in range(n_pop)
    ]


def decode_population_spikes(counts_by_window_and_population: list[int], n_pop: int) -> TorqueTrajectory:
    if len(counts_by_window_and_population) % n_pop != 0:
        raise ValueError("population spike count vector length is not divisible by n_pop")
    directions = population_directions(n_pop)
    window_count = len(counts_by_window_and_population) // n_pop
    q1: list[float] = []
    q2: list[float] = []
    for window in range(window_count):
        torque1 = 0.0
        torque2 = 0.0
        for pop in range(n_pop):
            count = counts_by_window_and_population[window * n_pop + pop]
            torque1 += directions[pop][0] * count
            torque2 += directions[pop][1] * count
        q1.append(torque1 * 1.0e-3)
        q2.append(torque2 * 1.0e-3)
    return TorqueTrajectory(q1=q1, q2=q2)


def simulate_cm_forward_exact(
    cfg: Stage2Config,
    ext_spikes: BinarySpikeSeries,
    weights: BinaryWeightWindows,
    *,
    max_windows: int | None = None,
) -> CmForwardResult:
    if ext_spikes.shape.rows != cfg.n_ext:
        raise ValueError("CM forward received unexpected external spike row count")
    if weights.shape.dim0 != cfg.n_ext or weights.shape.dim2 != cfg.n_cm:
        raise ValueError("CM forward received unexpected weight tensor shape")

    window_steps = int(10.0 / cfg.dt_ms)
    available_windows = min(weights.shape.dim1, ext_spikes.shape.cols // window_steps)
    window_count = available_windows if max_windows is None else min(available_windows, int(max_windows))
    step_count = window_count * window_steps
    refractory_steps = int(cfg.refractory_ms / cfg.dt_ms)
    neurons_per_pop = cfg.n_cm // cfg.n_pop
    weight_block = cfg.n_ext * cfg.n_cm

    vm = [cfg.v_reset] * cfg.n_cm
    s_ampa_ext = [0.0] * cfg.n_cm
    input_ext = [0.0] * cfg.n_cm
    last_spike_step = [-10**9] * cfg.n_cm
    population_counts = [0] * (window_count * cfg.n_pop)

    for step in range(step_count):
        current_window = step // window_steps
        for target in range(cfg.n_cm):
            input_ext[target] = 0.0

        spike_base = step * cfg.n_ext
        window_offset = current_window * weight_block
        active_sources = [source for source in range(cfg.n_ext) if ext_spikes.values[spike_base + source] != 0]
        for source in active_sources:
            for target in range(cfg.n_cm):
                input_ext[target] += weights.values[window_offset + target * cfg.n_ext + source]

        pop_base = current_window * cfg.n_pop
        for target in range(cfg.n_cm):
            s_ampa_ext[target] += (-s_ampa_ext[target] / cfg.tau_ampa + input_ext[target]) * cfg.dt_ms
            i_ex_ext = -cfg.g_ext_cm * s_ampa_ext[target] * (vm[target] - cfg.e_ampa)
            i_in = -cfg.g_l * (vm[target] - cfg.e_leak)
            dvm = (i_in + i_ex_ext) / cfg.cm_membrane_capacitance
            vm[target] += dvm * cfg.dt_ms

            v = vm[target]
            spike = 0
            if v >= cfg.v_th:
                v = cfg.v_spike
                vm[target] = cfg.v_reset
            if v > cfg.v_th:
                spike = 1
            if last_spike_step[target] >= step - refractory_steps:
                spike = 0
                v = cfg.v_reset
                vm[target] = cfg.v_reset
            if spike:
                last_spike_step[target] = step
                population_counts[pop_base + target // neurons_per_pop] += 1

    return CmForwardResult(population_counts=population_counts, simulated_windows=window_count)


def write_window_counts(path: Path, population_counts: list[int], n_pop: int) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("window")
        for pop in range(n_pop):
            handle.write(f"\tpop_{pop}")
        handle.write("\n")
        for window in range(len(population_counts) // n_pop):
            handle.write(str(window))
            for pop in range(n_pop):
                handle.write(f"\t{population_counts[window * n_pop + pop]}")
            handle.write("\n")


def write_torque_file(path: Path, torque: TorqueTrajectory) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("torque1\ttorque2\n")
        for torque1, torque2 in zip(torque.q1, torque.q2):
            handle.write(f"{torque1:.17g}\t{torque2:.17g}\n")


def write_summary(path: Path, desired_path: StrokePath, replay: SimulatedTrajectory) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"path_mse_x\t{mean_squared_error(desired_path.x, replay.hand_path.x):.17g}\n")
        handle.write(f"path_mse_y\t{mean_squared_error(desired_path.y, replay.hand_path.y):.17g}\n")


def replay_stage2_path(
    desired_path: StrokePath,
    decoded_torque: TorqueTrajectory,
    arm: ArmParameters,
    *,
    dt_seconds: float = 0.01,
) -> SimulatedTrajectory:
    desired_joints = build_desired_joint_trajectory(desired_path, arm, dt_seconds)
    return replay_forward_dynamics(desired_joints, decoded_torque, arm, dt_seconds)
