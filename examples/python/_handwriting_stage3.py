"""Handwriting stage-3 data and path helpers.

The full C++ stage-3 recurrent simulation is intentionally not reimplemented
here yet. These helpers migrate the stable file loading, mode parsing and
target-path construction pieces so the Python example can validate inputs and
prepare the same report artifacts before the heavy runtime is bound/vectorized.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import os

from _handwriting_stage1 import ArmParameters
from _handwriting_stage1 import JointTrajectory
from _handwriting_stage1 import SimulatedTrajectory
from _handwriting_stage1 import StrokePath
from _handwriting_stage1 import TorqueTrajectory
from _handwriting_stage1 import build_desired_joint_trajectory
from _handwriting_stage1 import hand_position
from _handwriting_stage1 import joint_acceleration
from _handwriting_stage1 import load_numeric_series
from _handwriting_stage1 import load_stroke_path_from_directory
from _handwriting_stage1 import mean_squared_error
from _handwriting_stage1 import replay_forward_dynamics
from _handwriting_stage2 import BinarySpikeSeries
from _handwriting_stage2 import BinaryWeightWindows
from _handwriting_stage2 import decode_population_spikes
from _handwriting_stage2 import load_binary_spike_series
from _handwriting_stage2 import load_binary_weight_windows
from _handwriting_stage2 import load_shape_2d


class Stage3Mode(Enum):
    TEST = "test"
    TEST_D = "test_d"
    TRAIN_G = "train_g"
    TRAIN_D = "train_d"


@dataclass(frozen=True)
class Stage3Config:
    n_ext: int = 100
    n_cm: int = 400
    n_bg: int = 10
    n_bg_total: int = 60
    n_sm_group: int = 10
    n_sm_total: int = 20
    n_mm: int = 500
    n_e: int = 1000
    n_i: int = 250
    n_sig: int = 20
    n_pop: int = 8
    n_rcv: int = 20
    total_time_ms: int = 9000
    dt_ms: float = 0.1
    verify_dt_s: float = 0.01
    refractory_ms: int = 2
    v_th: float = -50.0
    v_reset: float = -60.0
    v_spike: float = 20.0
    e_ampa: float = 0.0
    e_nmda: float = 0.0
    e_gaba: float = -70.0
    e_leak: float = -70.0
    cm: float = 500.0
    g_l: float = 25.0
    tau_ampa: float = 2.0
    tau_nmda: float = 100.0
    tau_gaba: float = 10.0
    g_ext_cm: float = 10.0
    g_bg_cm: float = 550.0
    g_cm_bg: float = 4.0
    g_sm_bg: float = 6.0
    g_mm_bg: float = 8.0
    g_ext_mm: float = 20.0
    g_e_mm: float = 20.0
    g_ext_e: float = 42.2
    g_ext_i: float = 40.0
    g_ee: float = 18.2
    g_ei: float = 18.2
    g_ie: float = 16.7
    g_ii: float = 16.7
    g_sig_e: float = 40.0
    g_cm_e: float = 40.0
    hebb_increment: float = 0.1


@dataclass(frozen=True)
class MatrixF32:
    rows: int = 0
    cols: int = 0
    values: array = field(default_factory=lambda: array("f"))


@dataclass(frozen=True)
class Stage3Data:
    sspk_ext: BinarySpikeSeries
    sspk_ext_e: BinarySpikeSeries
    sspk_ext_i: BinarySpikeSeries
    sspk_ext_mm: BinarySpikeSeries
    sspk_sig: BinarySpikeSeries
    sspk_sm: BinarySpikeSeries
    ww_extcm1: BinaryWeightWindows
    ww_extcm2: BinaryWeightWindows
    ww_extcm3: BinaryWeightWindows
    a1_flat: list[float]
    a2_flat: list[float]
    a3_flat: list[float]
    x1y1: StrokePath
    x2y2: StrokePath
    x3y3: StrokePath
    c_emm1: MatrixF32 = field(default_factory=MatrixF32)
    c_emm2: MatrixF32 = field(default_factory=MatrixF32)
    c_mmbg1: MatrixF32 = field(default_factory=MatrixF32)
    c_mmbg2: MatrixF32 = field(default_factory=MatrixF32)
    w_mmbg1: MatrixF32 = field(default_factory=MatrixF32)
    w_mmbg2: MatrixF32 = field(default_factory=MatrixF32)
    c_cm1e: MatrixF32 = field(default_factory=MatrixF32)
    c_cm2e: MatrixF32 = field(default_factory=MatrixF32)
    c_cm3e: MatrixF32 = field(default_factory=MatrixF32)
    c_sig_e: MatrixF32 = field(default_factory=MatrixF32)
    w_ee: MatrixF32 = field(default_factory=MatrixF32)
    c_emm_d: MatrixF32 = field(default_factory=MatrixF32)
    c_mmbg_d: MatrixF32 = field(default_factory=MatrixF32)
    w_mmbg_d: MatrixF32 = field(default_factory=MatrixF32)


@dataclass
class LayerStateNp:
    vm: np.ndarray
    last_spike_step: np.ndarray


@dataclass
class Stage3Result:
    qq: TorqueTrajectory
    replay: SimulatedTrajectory
    pop_spk: list[int]
    w_mmbg1: np.ndarray
    w_mmbg2: np.ndarray
    simulated_steps: int


class CppMt19937:
    """Small std::mt19937-compatible generator for baseline parity."""

    n = 624
    m = 397
    matrix_a = 0x9908B0DF
    upper_mask = 0x80000000
    lower_mask = 0x7FFFFFFF

    def __init__(self, seed: int = 0):
        self.state = [0] * self.n
        self.state[0] = int(seed) & 0xFFFFFFFF
        for index in range(1, self.n):
            previous = self.state[index - 1]
            self.state[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF
        self.index = self.n

    def _twist(self) -> None:
        for index in range(self.n):
            value = (self.state[index] & self.upper_mask) + (self.state[(index + 1) % self.n] & self.lower_mask)
            shifted = value >> 1
            if value & 1:
                shifted ^= self.matrix_a
            self.state[index] = self.state[(index + self.m) % self.n] ^ shifted
        self.index = 0

    def next_u32(self) -> int:
        if self.index >= self.n:
            self._twist()
        value = self.state[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF

    def uniform01_msvc_double(self) -> float:
        first = self.next_u32()
        second = self.next_u32()
        return (float(first) + float(second) * 4294967296.0) / 18446744073709551616.0


def parse_stage3_mode(text: str) -> Stage3Mode:
    normalized = text.strip().lower()
    if normalized == "test":
        return Stage3Mode.TEST
    if normalized in {"testd", "test_d"}:
        return Stage3Mode.TEST_D
    if normalized in {"traing", "train_g"}:
        return Stage3Mode.TRAIN_G
    if normalized in {"traind", "train_d"}:
        return Stage3Mode.TRAIN_D
    raise ValueError(f"unknown stage-3 mode: {text}")


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required input file not found: {path}")
    return path


def load_matrix_f32(data_path: Path, shape_path: Path) -> MatrixF32:
    shape = load_shape_2d(shape_path)
    values = array("f")
    with data_path.open("rb") as handle:
        values.fromfile(handle, shape.rows * shape.cols)
    if len(values) != shape.rows * shape.cols:
        raise ValueError(f"{data_path} has {len(values)} floats; expected {shape.rows * shape.cols}")
    return MatrixF32(rows=shape.rows, cols=shape.cols, values=values)


def load_optional_matrix(input_dir: Path, stem: str) -> MatrixF32:
    data_path = input_dir / f"{stem}.bin"
    shape_path = input_dir / f"{stem}.shape.txt"
    if not data_path.exists():
        return MatrixF32()
    return load_matrix_f32(require_path(data_path), require_path(shape_path))


def load_stage3_data(input_dir: Path, *, require_test_weights: bool = False, load_optional_weights: bool = True) -> Stage3Data:
    optional = load_optional_matrix if load_optional_weights else lambda _input_dir, _stem: MatrixF32()
    c_emm1 = optional(input_dir, "C_EMM1")
    c_emm2 = optional(input_dir, "C_EMM2")
    c_mmbg1 = optional(input_dir, "C_MMbg1")
    c_mmbg2 = optional(input_dir, "C_MMbg2")
    w_mmbg1 = optional(input_dir, "w_MMbg1_12")
    w_mmbg2 = optional(input_dir, "w_MMbg2_12")
    if require_test_weights:
        for name, matrix in (
            ("C_EMM1", c_emm1),
            ("C_EMM2", c_emm2),
            ("C_MMbg1", c_mmbg1),
            ("C_MMbg2", c_mmbg2),
            ("w_MMbg1_12", w_mmbg1),
            ("w_MMbg2_12", w_mmbg2),
        ):
            if not matrix.values:
                raise FileNotFoundError(f"required test matrix is missing: {name}")

    return Stage3Data(
        sspk_ext=load_binary_spike_series(require_path(input_dir / "sspk_ext.bin"), require_path(input_dir / "sspk_ext.shape.txt")),
        sspk_ext_e=load_binary_spike_series(require_path(input_dir / "sspk_extE.bin"), require_path(input_dir / "sspk_extE.shape.txt")),
        sspk_ext_i=load_binary_spike_series(require_path(input_dir / "sspk_extI.bin"), require_path(input_dir / "sspk_extI.shape.txt")),
        sspk_ext_mm=load_binary_spike_series(require_path(input_dir / "sspk_extMM.bin"), require_path(input_dir / "sspk_extMM.shape.txt")),
        sspk_sig=load_binary_spike_series(require_path(input_dir / "sspk_sig.bin"), require_path(input_dir / "sspk_sig.shape.txt")),
        sspk_sm=load_binary_spike_series(require_path(input_dir / "sspk_SM.bin"), require_path(input_dir / "sspk_SM.shape.txt")),
        ww_extcm1=load_binary_weight_windows(require_path(input_dir / "ww_extCM1.bin"), require_path(input_dir / "ww_extCM1.shape.txt")),
        ww_extcm2=load_binary_weight_windows(require_path(input_dir / "ww_extCM2.bin"), require_path(input_dir / "ww_extCM2.shape.txt")),
        ww_extcm3=load_binary_weight_windows(require_path(input_dir / "ww_extCM3.bin"), require_path(input_dir / "ww_extCM3.shape.txt")),
        a1_flat=load_numeric_series(require_path(input_dir / "A1.tsv")),
        a2_flat=load_numeric_series(require_path(input_dir / "A2.tsv")),
        a3_flat=load_numeric_series(require_path(input_dir / "A3.tsv")),
        x1y1=load_stroke_path_from_directory(input_dir, "1"),
        x2y2=load_stroke_path_from_directory(input_dir, "2"),
        x3y3=load_stroke_path_from_directory(input_dir, "3"),
        c_emm1=c_emm1,
        c_emm2=c_emm2,
        c_mmbg1=c_mmbg1,
        c_mmbg2=c_mmbg2,
        w_mmbg1=w_mmbg1,
        w_mmbg2=w_mmbg2,
        c_cm1e=optional(input_dir, "C_CM1E"),
        c_cm2e=optional(input_dir, "C_CM2E"),
        c_cm3e=optional(input_dir, "C_CM3E"),
        c_sig_e=optional(input_dir, "C_sigE"),
        w_ee=optional(input_dir, "w_EE"),
        c_emm_d=optional(input_dir, "C_EMM"),
        c_mmbg_d=optional(input_dir, "C_MMbg"),
        w_mmbg_d=optional(input_dir, "w_MMbg_13"),
    )


def build_padded_path(first: StrokePath, second: StrokePath, *, hold: int = 50) -> StrokePath:
    return StrokePath(
        x=[first.x[0]] * hold + list(first.x) + [second.x[0]] * hold + list(second.x),
        y=[first.y[0]] * hold + list(first.y) + [second.y[0]] * hold + list(second.y),
    )


def target_path_for_mode(data: Stage3Data, mode: Stage3Mode) -> StrokePath:
    if mode in (Stage3Mode.TRAIN_G, Stage3Mode.TEST):
        return build_padded_path(data.x1y1, data.x2y2)
    return build_padded_path(data.x1y1, data.x3y3)


def matrix_to_numpy(matrix: MatrixF32) -> np.ndarray:
    if not matrix.values:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(matrix.values, dtype=np.float32).reshape((matrix.cols, matrix.rows)).T.astype(np.float64, copy=False)


def spike_series_to_numpy(spikes: BinarySpikeSeries) -> np.ndarray:
    return np.frombuffer(spikes.values, dtype=np.uint8).reshape((spikes.shape.cols, spikes.shape.rows))


def weights_to_numpy(weights: BinaryWeightWindows) -> np.ndarray:
    return np.asarray(weights.values, dtype=np.float32).reshape((weights.shape.dim1, weights.shape.dim2, weights.shape.dim0)).astype(
        np.float64, copy=False
    )


def _update_ampa(state: np.ndarray, inputs: np.ndarray, cfg: Stage3Config) -> None:
    state += (-state / cfg.tau_ampa + inputs) * cfg.dt_ms


def _update_gaba(state: np.ndarray, inputs: np.ndarray, cfg: Stage3Config) -> None:
    state += (-state / cfg.tau_gaba + inputs) * cfg.dt_ms


def _update_nmda(state: np.ndarray, inputs: np.ndarray, cfg: Stage3Config) -> None:
    state += (-state / cfg.tau_nmda + 0.63 * (1.0 - state) * inputs) * cfg.dt_ms


def _apply_if_and_refractory(layer: LayerStateNp, step: int, cfg: Stage3Config) -> np.ndarray:
    spikes = layer.vm >= cfg.v_th
    layer.vm[spikes] = cfg.v_reset
    refractory = (layer.last_spike_step >= 0) & ((step - layer.last_spike_step) <= int(cfg.refractory_ms / cfg.dt_ms))
    spikes &= ~refractory
    layer.vm[refractory] = cfg.v_reset
    layer.last_spike_step[spikes] = step
    return spikes.astype(np.float64)


def _nmda_current(g: float, sa: np.ndarray, sn: np.ndarray, vm: np.ndarray, cfg: Stage3Config) -> np.ndarray:
    return -g * sa * (vm - cfg.e_ampa) - g * sn * (vm - cfg.e_nmda) / (1.0 + np.exp(-0.062 * vm / 3.57))


def connectivity_bg_cm(cfg: Stage3Config, which: int) -> np.ndarray:
    matrix = np.zeros((cfg.n_cm, cfg.n_bg_total), dtype=np.float64)
    for block in range(6):
        on = (
            (which == 1 and block in (1, 2, 4, 5))
            or (which == 2 and block in (0, 2, 3, 5))
            or (which == 3 and block in (0, 1, 3, 4))
        )
        if on:
            matrix[:, block * cfg.n_bg : (block + 1) * cfg.n_bg] = 1.0
    return matrix


def connectivity_cm_bg(cfg: Stage3Config, which: int) -> np.ndarray:
    matrix = np.zeros((cfg.n_bg_total, cfg.n_cm), dtype=np.float64)
    for block in range(6):
        on = (
            (which == 1 and block in (0, 3))
            or (which == 2 and block in (1, 4))
            or (which == 3 and block in (2, 5))
        )
        if on:
            matrix[block * cfg.n_bg : (block + 1) * cfg.n_bg, :] = 1.0
    return matrix


def connectivity_sm_bg(cfg: Stage3Config) -> np.ndarray:
    matrix = np.zeros((cfg.n_bg_total, cfg.n_sm_total), dtype=np.float64)
    for row in range(cfg.n_bg):
        if row < cfg.n_sm_group:
            matrix[0 * cfg.n_bg + row, row] = 1.0
            matrix[1 * cfg.n_bg + row, row] = 1.0
            matrix[2 * cfg.n_bg + row, row] = 1.0
            matrix[3 * cfg.n_bg + row, cfg.n_sm_group + row] = 1.0
            matrix[4 * cfg.n_bg + row, cfg.n_sm_group + row] = 1.0
            matrix[5 * cfg.n_bg + row, cfg.n_sm_group + row] = 1.0
    return matrix


def structured_connectivity_ee(n_e: int) -> np.ndarray:
    idx = np.arange(n_e, dtype=np.float64)
    return np.exp(-((idx[:, None] - idx[None, :]) ** 2) / 3000.0)


def connectivity_cm_e(cfg: Stage3Config, center: int) -> np.ndarray:
    rng = CppMt19937(0)
    matrix = np.zeros((cfg.n_e, cfg.n_cm), dtype=np.float32)
    begin = center - cfg.n_rcv // 2
    for local_row in range(cfg.n_rcv):
        row = begin + local_row
        for col in range(cfg.n_cm):
            if rng.uniform01_msvc_double() <= 0.5:
                matrix[row, col] = 1.0
    return matrix


def connectivity_signal_e(cfg: Stage3Config) -> np.ndarray:
    rng = CppMt19937(1)
    matrix = np.zeros((cfg.n_e, cfg.n_sig), dtype=np.float32)
    begin = 800 - cfg.n_rcv // 2
    for local_row in range(cfg.n_rcv):
        row = begin + local_row
        for col in range(cfg.n_sig):
            if rng.uniform01_msvc_double() <= 0.5:
                matrix[row, col] = 1.0
    return matrix


def random_binary_connectivity(rows: int, cols: int, probability: float, rng: CppMt19937) -> np.ndarray:
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for col in range(cols):
        for row in range(rows):
            if rng.uniform01_msvc_double() <= probability:
                matrix[row, col] = 1.0
    return matrix


def expand_top_half_mask(half_mask: np.ndarray, total_rows: int) -> np.ndarray:
    matrix = np.zeros((total_rows, half_mask.shape[1]), dtype=np.float32)
    matrix[: half_mask.shape[0], :] = half_mask
    return matrix


def expand_bottom_half_mask(half_mask: np.ndarray, total_rows: int) -> np.ndarray:
    matrix = np.zeros((total_rows, half_mask.shape[1]), dtype=np.float32)
    row_begin = total_rows // 2
    matrix[row_begin : row_begin + half_mask.shape[0], :] = half_mask
    return matrix


def build_window_index_sequence(window_count: int) -> list[int]:
    sequence = [0] * 50 + list(range(400)) + [0] * 50 + list(range(400))
    return sequence[:window_count]


def _cm_ext_step(mode: Stage3Mode, channel: int, step: int, is_test: bool) -> int:
    if is_test and mode == Stage3Mode.TEST_D:
        if channel == 0 and 5000 <= step < 45000:
            return step - 5000
        if channel == 2 and 50000 <= step < 90000:
            return step - 50000
        return -1
    if is_test:
        if 5000 <= step < 45000:
            return step - 5000
        if 50000 <= step < 90000:
            return step - 50000
        return -1
    if channel == 0 and step >= 5000:
        return step - 5000
    if mode == Stage3Mode.TRAIN_G and channel == 1 and step >= 50000:
        return step - 50000
    if mode == Stage3Mode.TRAIN_D and channel == 2 and step >= 50000:
        return step - 50000
    return -1


def _cm_external(ext: np.ndarray, mode: Stage3Mode, channel: int, step: int, is_test: bool) -> np.ndarray:
    local = _cm_ext_step(mode, channel, step, is_test)
    if local < 0 or local >= ext.shape[0]:
        return np.zeros(ext.shape[1], dtype=np.float64)
    return ext[local].astype(np.float64, copy=False)


def _update_cm(
    cfg: Stage3Config,
    weights: np.ndarray,
    window_index: int,
    c_bgcm: np.ndarray,
    ext_spike: np.ndarray,
    bg_spike: np.ndarray,
    step: int,
    layer: LayerStateNp,
    s_state: np.ndarray,
) -> np.ndarray:
    s_ampa_ext = s_state[: cfg.n_cm]
    s_gaba_bg = s_state[cfg.n_cm :]
    input_ext = weights[window_index] @ ext_spike
    input_bg = c_bgcm @ bg_spike
    _update_ampa(s_ampa_ext, input_ext, cfg)
    _update_gaba(s_gaba_bg, input_bg, cfg)
    vm = layer.vm
    i_ex_ext = -cfg.g_ext_cm * s_ampa_ext * (vm - cfg.e_ampa)
    i_ex_bg = -cfg.g_bg_cm * s_gaba_bg * (vm - cfg.e_gaba)
    i_in = -cfg.g_l * (vm - cfg.e_leak)
    layer.vm += (i_in + i_ex_ext + i_ex_bg) / cfg.cm * cfg.dt_ms
    return _apply_if_and_refractory(layer, step, cfg)


def _update_bg(
    cfg: Stage3Config,
    matrices: dict[str, np.ndarray],
    spikes: dict[str, np.ndarray],
    step: int,
    layer: LayerStateNp,
    s_state: np.ndarray,
) -> np.ndarray:
    n = cfg.n_bg_total
    states = [s_state[i * n : (i + 1) * n] for i in range(12)]
    inputs = [
        matrices["c_cm1bg"] @ spikes["cm1"],
        matrices["c_cm2bg"] @ spikes["cm2"],
        matrices["c_cm3bg"] @ spikes["cm3"],
        matrices["c_smbg"] @ spikes["sm"],
        (matrices["c_mmbg1"] * matrices["w_mmbg1"]) @ spikes["mm1"],
        (matrices["c_mmbg2"] * matrices["w_mmbg2"]) @ spikes["mm2"],
    ]
    for index, values in enumerate(inputs):
        _update_ampa(states[index * 2], values, cfg)
        _update_nmda(states[index * 2 + 1], values, cfg)
    vm = layer.vm
    i_ex = (
        _nmda_current(cfg.g_cm_bg, states[0], states[1], vm, cfg)
        + _nmda_current(cfg.g_cm_bg, states[2], states[3], vm, cfg)
        + _nmda_current(cfg.g_cm_bg, states[4], states[5], vm, cfg)
        + _nmda_current(cfg.g_sm_bg, states[6], states[7], vm, cfg)
        + _nmda_current(cfg.g_mm_bg, states[8], states[9], vm, cfg)
        + _nmda_current(cfg.g_mm_bg, states[10], states[11], vm, cfg)
    )
    i_in = -cfg.g_l * (vm - cfg.e_leak)
    layer.vm += (i_in + i_ex) / cfg.cm * cfg.dt_ms
    return _apply_if_and_refractory(layer, step, cfg)


def _update_mm(
    cfg: Stage3Config,
    c_emm: np.ndarray,
    ext_spike: np.ndarray,
    e_spike: np.ndarray,
    step: int,
    layer: LayerStateNp,
    s_state: np.ndarray,
) -> np.ndarray:
    n = cfg.n_mm
    s_ampa_ext = s_state[:n]
    s_ampa_e = s_state[n : 2 * n]
    s_nmda_e = s_state[2 * n :]
    input_e = c_emm @ e_spike
    _update_ampa(s_ampa_ext, ext_spike, cfg)
    _update_ampa(s_ampa_e, input_e, cfg)
    _update_nmda(s_nmda_e, input_e, cfg)
    vm = layer.vm
    i_ex_ext = -cfg.g_ext_mm * s_ampa_ext * (vm - cfg.e_ampa)
    i_ex_e = _nmda_current(cfg.g_e_mm, s_ampa_e, s_nmda_e, vm, cfg)
    i_in = -cfg.g_l * (vm - cfg.e_leak)
    layer.vm += (i_in + i_ex_ext + i_ex_e) / cfg.cm * cfg.dt_ms
    return _apply_if_and_refractory(layer, step, cfg)


def _update_e(
    cfg: Stage3Config,
    matrices: dict[str, np.ndarray],
    spikes: dict[str, np.ndarray],
    step: int,
    layer: LayerStateNp,
    s_state: np.ndarray,
) -> np.ndarray:
    n = cfg.n_e
    states = [s_state[i * n : (i + 1) * n] for i in range(11)]
    input_ext = spikes["ext_e"]
    input_e = matrices["w_ee"] @ spikes["e"]
    input_i = np.full(n, float(np.sum(spikes["i"])), dtype=np.float64)
    input_sig = matrices["c_sig_e"] @ spikes["sig"]
    input_cm1 = matrices["c_cm1e"] @ spikes["cm1"]
    input_cm2 = matrices["c_cm2e"] @ spikes["cm2"]
    input_cm3 = matrices["c_cm3e"] @ spikes["cm3"]
    _update_ampa(states[0], input_ext, cfg)
    _update_ampa(states[1], input_e, cfg)
    _update_nmda(states[2], input_e, cfg)
    _update_gaba(states[3], input_i, cfg)
    _update_ampa(states[4], input_sig, cfg)
    _update_ampa(states[5], input_cm1, cfg)
    _update_nmda(states[6], input_cm1, cfg)
    _update_ampa(states[7], input_cm2, cfg)
    _update_nmda(states[8], input_cm2, cfg)
    _update_ampa(states[9], input_cm3, cfg)
    _update_nmda(states[10], input_cm3, cfg)
    vm = layer.vm
    i_ex = (
        -cfg.g_ext_e * states[0] * (vm - cfg.e_ampa)
        + _nmda_current(cfg.g_ee, states[1], states[2], vm, cfg)
        - cfg.g_ie * states[3] * (vm - cfg.e_gaba)
        - cfg.g_sig_e * states[4] * (vm - cfg.e_ampa)
        + _nmda_current(cfg.g_cm_e, states[5], states[6], vm, cfg)
        + _nmda_current(cfg.g_cm_e, states[7], states[8], vm, cfg)
        + _nmda_current(cfg.g_cm_e, states[9], states[10], vm, cfg)
    )
    i_in = -cfg.g_l * (vm - cfg.e_leak)
    layer.vm += (i_in + i_ex) / cfg.cm * cfg.dt_ms
    return _apply_if_and_refractory(layer, step, cfg)


def _update_i(
    cfg: Stage3Config,
    ext_i: np.ndarray,
    e_spike: np.ndarray,
    i_spike: np.ndarray,
    step: int,
    layer: LayerStateNp,
    s_state: np.ndarray,
) -> np.ndarray:
    n = cfg.n_i
    states = [s_state[i * n : (i + 1) * n] for i in range(4)]
    input_e = np.full(n, float(np.sum(e_spike)), dtype=np.float64)
    input_i = np.full(n, float(np.sum(i_spike)), dtype=np.float64)
    _update_ampa(states[0], ext_i, cfg)
    _update_ampa(states[1], input_e, cfg)
    _update_nmda(states[2], input_e, cfg)
    _update_gaba(states[3], input_i, cfg)
    vm = layer.vm
    i_ex = (
        -cfg.g_ext_i * states[0] * (vm - cfg.e_ampa)
        + _nmda_current(cfg.g_ei, states[1], states[2], vm, cfg)
        - cfg.g_ii * states[3] * (vm - cfg.e_gaba)
    )
    i_in = -cfg.g_l * (vm - cfg.e_leak)
    layer.vm += (i_in + i_ex) / cfg.cm * cfg.dt_ms
    return _apply_if_and_refractory(layer, step, cfg)


def _hebb_update_strict_matlab(
    weights: np.ndarray,
    mask: np.ndarray,
    mm_window_counts: np.ndarray,
    bg_window_counts: np.ndarray,
    increment: float,
    rng: CppMt19937,
) -> None:
    row_count, col_count = weights.shape
    valid_rows = min(row_count, len(mm_window_counts), len(bg_window_counts))
    for row in range(row_count):
        row_active = row < valid_rows and mm_window_counts[row] > 0 and bg_window_counts[row] > 0
        if not row_active:
            continue
        for col in range(col_count):
            if mask[row, col] == 1.0:
                weights[row, col] = np.float32(float(weights[row, col]) + float(increment) * rng.uniform01_msvc_double())
    np.clip(weights, 0.0, 1.0, out=weights)


def replay_forward_dynamics_with_reset(
    torques: TorqueTrajectory,
    params: ArmParameters,
    dt_seconds: float,
    initial_theta1: float,
    initial_theta2: float,
    reset_after_step: int,
    reset_theta1: float,
    reset_theta2: float,
) -> SimulatedTrajectory:
    step_count = len(torques.q1)
    theta1 = [initial_theta1] * (step_count + 1)
    theta2 = [initial_theta2] * (step_count + 1)
    dtheta1 = [0.0] * (step_count + 1)
    dtheta2 = [0.0] * (step_count + 1)
    ddtheta1 = [0.0] * step_count
    ddtheta2 = [0.0] * step_count
    for index in range(step_count):
        acc1, acc2 = joint_acceleration(theta2[index], dtheta1[index], dtheta2[index], torques.q1[index], torques.q2[index], params)
        ddtheta1[index] = acc1
        ddtheta2[index] = acc2
        dtheta1[index + 1] = dtheta1[index] + acc1 * dt_seconds
        dtheta2[index + 1] = dtheta2[index] + acc2 * dt_seconds
        theta1[index + 1] = theta1[index] + dtheta1[index] * dt_seconds
        theta2[index + 1] = theta2[index] + dtheta2[index] * dt_seconds
        if index + 1 == reset_after_step:
            theta1[index + 1] = reset_theta1
            theta2[index + 1] = reset_theta2
    return SimulatedTrajectory(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2, hand_position(theta1, theta2, params))


def first_joint_pair(flat: list[float]) -> tuple[float, float]:
    if len(flat) < 2 or len(flat) % 2 != 0:
        raise ValueError("joint angle text export must contain an even number of values")
    stride = len(flat) // 2
    return flat[0], flat[stride]


def run_stage3_numpy(
    cfg: Stage3Config,
    data: Stage3Data,
    mode: Stage3Mode,
    *,
    max_steps: int | None = None,
) -> Stage3Result:
    is_test = mode in (Stage3Mode.TEST, Stage3Mode.TEST_D)
    cfg_local = cfg
    if mode == Stage3Mode.TEST_D:
        cfg_local = Stage3Config(**{**cfg.__dict__, "g_sm_bg": 10.0, "g_mm_bg": 10.0})

    step_count = int(cfg.total_time_ms / cfg.dt_ms)
    if max_steps is not None:
        step_count = min(step_count, int(max_steps))
    window_steps = int(10.0 / cfg.dt_ms)
    window_count = step_count // window_steps
    window_sequence = build_window_index_sequence(window_count)

    ext = spike_series_to_numpy(data.sspk_ext)
    ext_e = spike_series_to_numpy(data.sspk_ext_e)
    ext_i = spike_series_to_numpy(data.sspk_ext_i)
    ext_mm = spike_series_to_numpy(data.sspk_ext_mm)
    sig = spike_series_to_numpy(data.sspk_sig)
    sm = spike_series_to_numpy(data.sspk_sm)
    ww1 = weights_to_numpy(data.ww_extcm1)
    ww2 = weights_to_numpy(data.ww_extcm2)
    ww3 = weights_to_numpy(data.ww_extcm3)
    rng = CppMt19937(0)

    matrices: dict[str, np.ndarray] = {
        "c_bgcm1": connectivity_bg_cm(cfg, 1),
        "c_bgcm2": connectivity_bg_cm(cfg, 2),
        "c_bgcm3": connectivity_bg_cm(cfg, 3),
        "c_cm1bg": connectivity_cm_bg(cfg, 1),
        "c_cm2bg": connectivity_cm_bg(cfg, 2),
        "c_cm3bg": connectivity_cm_bg(cfg, 3),
        "c_smbg": connectivity_sm_bg(cfg),
        "w_ee": matrix_to_numpy(data.w_ee) if data.w_ee.values else structured_connectivity_ee(cfg.n_e),
        "c_cm1e": matrix_to_numpy(data.c_cm1e) if data.c_cm1e.values else connectivity_cm_e(cfg, 200),
        "c_cm2e": matrix_to_numpy(data.c_cm2e) if data.c_cm2e.values else connectivity_cm_e(cfg, 400),
        "c_cm3e": matrix_to_numpy(data.c_cm3e) if data.c_cm3e.values else connectivity_cm_e(cfg, 600),
        "c_sig_e": matrix_to_numpy(data.c_sig_e) if data.c_sig_e.values else connectivity_signal_e(cfg),
    }
    if is_test:
        matrices["c_emm1"] = matrix_to_numpy(data.c_emm1)
        matrices["c_emm2"] = matrix_to_numpy(data.c_emm2)
        matrices["c_mmbg1"] = matrix_to_numpy(data.c_mmbg1)
        matrices["c_mmbg2"] = matrix_to_numpy(data.c_mmbg2)
        matrices["w_mmbg1"] = matrix_to_numpy(data.w_mmbg1)
        matrices["w_mmbg2"] = matrix_to_numpy(data.w_mmbg2)
    else:
        matrices["c_emm1"] = random_binary_connectivity(cfg.n_mm, cfg.n_e, 0.1, rng)
        matrices["c_emm2"] = random_binary_connectivity(cfg.n_mm, cfg.n_e, 0.1, rng)
        matrices["c_mmbg1"] = expand_top_half_mask(
            random_binary_connectivity(cfg.n_bg_total // 2, cfg.n_mm, 0.1, rng),
            cfg.n_bg_total,
        )
        matrices["c_mmbg2"] = expand_bottom_half_mask(
            random_binary_connectivity(cfg.n_bg_total // 2, cfg.n_mm, 0.1, rng),
            cfg.n_bg_total,
        )
        matrices["w_mmbg1"] = np.zeros((cfg.n_bg_total, cfg.n_mm), dtype=np.float32)
        matrices["w_mmbg2"] = np.zeros((cfg.n_bg_total, cfg.n_mm), dtype=np.float32)
    if mode == Stage3Mode.TEST_D and data.c_emm_d.values and data.c_mmbg_d.values and data.w_mmbg_d.values:
        c_emm_d = matrix_to_numpy(data.c_emm_d)
        c_mmbg_d = matrix_to_numpy(data.c_mmbg_d)
        w_mmbg_d = matrix_to_numpy(data.w_mmbg_d)
        matrices["c_emm1"] = c_emm_d[: cfg.n_mm, :]
        matrices["c_emm2"] = c_emm_d[cfg.n_mm : 2 * cfg.n_mm, :]
        matrices["c_mmbg1"] = c_mmbg_d[:, : cfg.n_mm]
        matrices["c_mmbg2"] = c_mmbg_d[:, cfg.n_mm : 2 * cfg.n_mm]
        matrices["w_mmbg1"] = np.where(w_mmbg_d[:, : cfg.n_mm] < 1.0, 0.0, w_mmbg_d[:, : cfg.n_mm])
        matrices["w_mmbg2"] = np.where(w_mmbg_d[:, cfg.n_mm : 2 * cfg.n_mm] < 1.0, 0.0, w_mmbg_d[:, cfg.n_mm : 2 * cfg.n_mm])

    def init_layer(count: int) -> LayerStateNp:
        return LayerStateNp(np.full(count, cfg.v_reset, dtype=np.float64), np.full(count, -1_000_000, dtype=np.int64))

    cm1 = init_layer(cfg.n_cm)
    cm2 = init_layer(cfg.n_cm)
    cm3 = init_layer(cfg.n_cm)
    bg = init_layer(cfg.n_bg_total)
    mm1 = init_layer(cfg.n_mm)
    mm2 = init_layer(cfg.n_mm)
    e = init_layer(cfg.n_e)
    i = init_layer(cfg.n_i)
    s_cm1 = np.zeros(2 * cfg.n_cm, dtype=np.float64)
    s_cm2 = np.zeros(2 * cfg.n_cm, dtype=np.float64)
    s_cm3 = np.zeros(2 * cfg.n_cm, dtype=np.float64)
    s_bg = np.zeros(12 * cfg.n_bg_total, dtype=np.float64)
    s_mm1 = np.zeros(3 * cfg.n_mm, dtype=np.float64)
    s_mm2 = np.zeros(3 * cfg.n_mm, dtype=np.float64)
    s_e = np.zeros(11 * cfg.n_e, dtype=np.float64)
    s_i = np.zeros(4 * cfg.n_i, dtype=np.float64)

    spk_bg = np.zeros(cfg.n_bg_total, dtype=np.float64)
    spk_mm1 = np.zeros(cfg.n_mm, dtype=np.float64)
    spk_mm2 = np.zeros(cfg.n_mm, dtype=np.float64)
    spk_e = np.zeros(cfg.n_e, dtype=np.float64)
    spk_i = np.zeros(cfg.n_i, dtype=np.float64)
    pop_counts = np.zeros((window_count, cfg.n_pop), dtype=np.int64)
    neurons_per_pop = cfg.n_cm // cfg.n_pop
    mm1_window_counts = np.zeros(cfg.n_mm, dtype=np.int64)
    mm2_window_counts = np.zeros(cfg.n_mm, dtype=np.int64)
    bg_window_counts = np.zeros(cfg.n_bg_total, dtype=np.int64)
    diag_dir_text = os.environ.get("HANDWRITING_STAGE3_DIAG_DIR")
    write_train_diag = (not is_test) and bool(diag_dir_text)
    train_updates1 = np.zeros((cfg.n_bg_total, cfg.n_mm), dtype=np.int32)
    train_updates2 = np.zeros((cfg.n_bg_total, cfg.n_mm), dtype=np.int32)

    for step in range(step_count):
        window = min(window_count - 1, step // window_steps)
        weight_window = window_sequence[window]
        spk_cm1 = _update_cm(cfg_local, ww1, weight_window, matrices["c_bgcm1"], _cm_external(ext, mode, 0, step, is_test), spk_bg, step, cm1, s_cm1)
        spk_cm2 = _update_cm(cfg_local, ww2, weight_window, matrices["c_bgcm2"], _cm_external(ext, mode, 1, step, is_test), spk_bg, step, cm2, s_cm2)
        spk_cm3 = _update_cm(cfg_local, ww3, weight_window, matrices["c_bgcm3"], _cm_external(ext, mode, 2, step, is_test), spk_bg, step, cm3, s_cm3)
        spk_sm = sm[step].astype(np.float64, copy=False)
        spk_sig = sig[step].astype(np.float64, copy=False)
        ext_mm1 = ext_mm[step, : cfg.n_mm].astype(np.float64, copy=False)
        ext_mm2 = ext_mm[step, cfg.n_mm : 2 * cfg.n_mm].astype(np.float64, copy=False)
        ext_e_step = ext_e[step].astype(np.float64, copy=False)
        ext_i_step = ext_i[step].astype(np.float64, copy=False)
        spk_bg = _update_bg(
            cfg_local,
            matrices,
            {"cm1": spk_cm1, "cm2": spk_cm2, "cm3": spk_cm3, "sm": spk_sm, "mm1": spk_mm1, "mm2": spk_mm2},
            step,
            bg,
            s_bg,
        )
        spk_mm1 = _update_mm(cfg_local, matrices["c_emm1"], ext_mm1, spk_e, step, mm1, s_mm1)
        spk_mm2 = _update_mm(cfg_local, matrices["c_emm2"], ext_mm2, spk_e, step, mm2, s_mm2)
        spk_e = _update_e(
            cfg_local,
            matrices,
            {
                "ext_e": ext_e_step,
                "e": spk_e,
                "i": spk_i,
                "sig": spk_sig,
                "cm1": spk_cm1,
                "cm2": spk_cm2,
                "cm3": spk_cm3,
            },
            step,
            e,
            s_e,
        )
        spk_i = _update_i(cfg_local, ext_i_step, spk_e, spk_i, step, i, s_i)
        if window_count:
            cm_total = (spk_cm1 + spk_cm2 + spk_cm3).reshape(cfg.n_pop, neurons_per_pop).sum(axis=1)
            pop_counts[window] += cm_total.astype(np.int64)
        if not is_test:
            mm1_window_counts += spk_mm1.astype(np.int64)
            mm2_window_counts += spk_mm2.astype(np.int64)
            bg_window_counts += spk_bg.astype(np.int64)
            if (step + 1) % window_steps == 0:
                if write_train_diag:
                    for row in range(cfg.n_bg_total):
                        if row < len(mm1_window_counts) and row < len(bg_window_counts):
                            if mm1_window_counts[row] > 0 and bg_window_counts[row] > 0:
                                train_updates1[row] += (matrices["c_mmbg1"][row] == 1.0).astype(np.int32)
                            if mm2_window_counts[row] > 0 and bg_window_counts[row] > 0:
                                train_updates2[row] += (matrices["c_mmbg2"][row] == 1.0).astype(np.int32)
                _hebb_update_strict_matlab(
                    matrices["w_mmbg1"],
                    matrices["c_mmbg1"],
                    mm1_window_counts,
                    bg_window_counts,
                    cfg_local.hebb_increment,
                    rng,
                )
                _hebb_update_strict_matlab(
                    matrices["w_mmbg2"],
                    matrices["c_mmbg2"],
                    mm2_window_counts,
                    bg_window_counts,
                    cfg_local.hebb_increment,
                    rng,
                )
                mm1_window_counts.fill(0)
                mm2_window_counts.fill(0)
                bg_window_counts.fill(0)

    if write_train_diag:
        diag_dir = Path(diag_dir_text)
        diag_dir.mkdir(parents=True, exist_ok=True)
        _write_int_matrix(diag_dir / "w_mmbg1_update_counts.tsv", train_updates1)
        _write_int_matrix(diag_dir / "w_mmbg2_update_counts.tsv", train_updates2)

    pop_spk = pop_counts.reshape(-1).astype(int).tolist()
    qq = decode_population_spikes(pop_spk, cfg.n_pop)
    start_path = StrokePath([data.x1y1.x[0]], [data.x1y1.y[0]])
    start_joint = build_desired_joint_trajectory(start_path, ArmParameters(), cfg.verify_dt_s)
    if mode == Stage3Mode.TEST_D:
        reset_theta1, reset_theta2 = first_joint_pair(data.a3_flat)
        replay = replay_forward_dynamics_with_reset(
            qq, ArmParameters(), cfg.verify_dt_s, start_joint.theta1[0], start_joint.theta2[0], 451, reset_theta1, reset_theta2
        )
    else:
        desired = JointTrajectory(
            theta1=[start_joint.theta1[0]] * (len(qq.q1) + 1),
            theta2=[start_joint.theta2[0]] * (len(qq.q2) + 1),
            dtheta1=[],
            dtheta2=[],
            ddtheta1=[],
            ddtheta2=[],
        )
        replay = replay_forward_dynamics(desired, qq, ArmParameters(), cfg.verify_dt_s)
    return Stage3Result(qq, replay, pop_spk, matrices["w_mmbg1"], matrices["w_mmbg2"], step_count)


def write_stage3_outputs(output_dir: Path, result: Stage3Result, target_path: StrokePath) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "QQ.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("torque1\ttorque2\n")
        for q1, q2 in zip(result.qq.q1, result.qq.q2):
            handle.write(f"{q1:.17g}\t{q2:.17g}\n")
    with (output_dir / "pop_spk.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("window" + "".join(f"\tpop_{i}" for i in range(8)) + "\n")
        for window in range(len(result.pop_spk) // 8):
            values = result.pop_spk[window * 8 : (window + 1) * 8]
            handle.write(str(window) + "".join(f"\t{value}" for value in values) + "\n")
    _write_path(output_dir / "replay_path.tsv", result.replay.hand_path)
    _write_path(output_dir / "target_path.tsv", target_path)
    count_x = min(len(target_path.x), len(result.replay.hand_path.x))
    count_y = min(len(target_path.y), len(result.replay.hand_path.y))
    with (output_dir / "summary.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("metric\tvalue\n")
        if count_x and count_y:
            handle.write(
                f"path_mse_x\t{mean_squared_error(target_path.x[:count_x], result.replay.hand_path.x[:count_x]):.17g}\n"
            )
            handle.write(
                f"path_mse_y\t{mean_squared_error(target_path.y[:count_y], result.replay.hand_path.y[:count_y]):.17g}\n"
            )
        else:
            handle.write("path_mse_x\tNA\npath_mse_y\tNA\n")
    if result.w_mmbg1.size:
        _write_matrix(output_dir / "w_mmbg1_final.tsv", result.w_mmbg1)
    if result.w_mmbg2.size:
        _write_matrix(output_dir / "w_mmbg2_final.tsv", result.w_mmbg2)


def _write_path(path: Path, stroke_path: StrokePath) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("x\ty\n")
        for x, y in zip(stroke_path.x, stroke_path.y):
            handle.write(f"{x:.17g}\t{y:.17g}\n")


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in range(matrix.shape[0]):
            handle.write("\t".join(f"{float(value):.6g}" for value in matrix[row]) + "\n")


def _write_int_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in range(matrix.shape[0]):
            handle.write("\t".join(str(int(value)) for value in matrix[row]) + "\n")
