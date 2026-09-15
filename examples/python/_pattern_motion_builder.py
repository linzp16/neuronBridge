"""Reusable builder for the migrated pattern-motion example."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct

import neuronbridge as nb


DIRS = 8
PLAID_ANGLE_DEG = 120.0
SPATIAL_FREQ = 0.1205
TEMPORAL_FREQ = 0.1808
STIMULUS_CONTRAST = 0.30
V1_SPEED = 1.5

V1_CDS_EXC_WEIGHT = 1.5
V1_CDS_INH_WEIGHT = 0.20
PDS_TO_LIP_WEIGHT = 0.060
LIP_INHIBITION_WEIGHT = 0.05
CDS_PDS_ALPHA = 0.068
CDS_PDS_POOL_SIGMA_PIXELS = 3.0
CDS_PDS_WEIGHT_EPSILON = 0.001
PDS_FS_TO_PDS_INH_WEIGHT = 0.02
PDS_FS_EXC_WEIGHT = 0.32
PDS_TUNED_DIR_SIGMA_DEG = 20.0
PDS_TUNED_LOC_SIGMA_PIXELS = 2.0
PDS_TUNED_WEIGHT_EPSILON = 0.002


@dataclass(frozen=True, slots=True)
class PatternMotionSpec:
    width: int = 8
    height: int = 8
    frames_per_direction: int = 4
    v1_update_interval_steps: int = 40
    pds_grid_width: int = 4
    pds_grid_height: int = 4
    lip_grid_width: int = 4
    lip_grid_height: int = 2
    dense_name: str = "pattern_motion_dense"

    @classmethod
    def full(cls) -> "PatternMotionSpec":
        return cls(
            width=32,
            height=32,
            frames_per_direction=10,
            v1_update_interval_steps=200,
            pds_grid_width=10,
            pds_grid_height=10,
            lip_grid_width=10,
            lip_grid_height=5,
        )

    @property
    def v1_count(self) -> int:
        return self.width * self.height * DIRS

    @property
    def cds_count(self) -> int:
        return self.v1_count

    @property
    def pds_pool_size(self) -> int:
        return self.pds_grid_width * self.pds_grid_height

    @property
    def pds_count(self) -> int:
        return DIRS * self.pds_pool_size

    @property
    def pds_fs_count(self) -> int:
        return self.pds_count

    @property
    def lip_pool_size(self) -> int:
        return self.lip_grid_width * self.lip_grid_height

    @property
    def lip_count(self) -> int:
        return DIRS * self.lip_pool_size

    @property
    def v1_base(self) -> int:
        return 0

    @property
    def cds_base(self) -> int:
        return self.v1_base + self.v1_count

    @property
    def pds_base(self) -> int:
        return self.cds_base + self.cds_count

    @property
    def pds_fs_base(self) -> int:
        return self.pds_base + self.pds_count

    @property
    def lip_base(self) -> int:
        return self.pds_fs_base + self.pds_fs_count

    @property
    def total_dense_neurons(self) -> int:
        return self.v1_count + self.cds_count + self.pds_count + self.pds_fs_count + self.lip_count

    @property
    def simulation_steps(self) -> int:
        return self.frames_per_direction * self.v1_update_interval_steps


def v1_neuron_id(spec: PatternMotionSpec, direction: int, spatial_index: int) -> int:
    return spec.v1_base + direction * (spec.width * spec.height) + spatial_index


def cds_neuron_id(spec: PatternMotionSpec, direction: int, spatial_index: int) -> int:
    return spec.cds_base + spatial_index * DIRS + direction


def pds_neuron_id(spec: PatternMotionSpec, direction: int, local_index: int) -> int:
    return spec.pds_base + direction * spec.pds_pool_size + local_index


def pds_fs_neuron_id(spec: PatternMotionSpec, direction: int, local_index: int) -> int:
    return spec.pds_fs_base + direction * spec.pds_pool_size + local_index


def lip_neuron_id(spec: PatternMotionSpec, direction: int, local_index: int) -> int:
    return spec.lip_base + direction * spec.lip_pool_size + local_index


def original_id(dense_local_id: int) -> int:
    return dense_local_id + 1


def direction_vector(direction_deg: float) -> tuple[float, float]:
    theta = direction_deg * math.pi / 180.0
    return math.sin(theta), -math.cos(theta)


def make_grating_frame(spec: PatternMotionSpec, direction_deg: float, frame_index: int, contrast: float) -> bytes:
    dx, dy = direction_vector(direction_deg)
    frame = bytearray(spec.width * spec.height)
    for y in range(spec.height):
        for x in range(spec.width):
            phase = 2.0 * math.pi * (SPATIAL_FREQ * (x * dx + y * dy) - TEMPORAL_FREQ * frame_index)
            value = 0.5 + 0.5 * contrast * math.sin(phase)
            frame[y * spec.width + x] = int(round(max(0.0, min(255.0, value * 255.0))))
    return bytes(frame)


def make_plaid_frame(spec: PatternMotionSpec, direction_deg: float, frame_index: int, contrast: float) -> bytes:
    first = make_grating_frame(spec, direction_deg + PLAID_ANGLE_DEG * 0.5, frame_index, contrast * 0.5)
    second = make_grating_frame(spec, direction_deg - PLAID_ANGLE_DEG * 0.5, frame_index, contrast * 0.5)
    return bytes(max(0, min(255, a + b - 128)) for a, b in zip(first, second))


def write_stimulus_file(path: Path, spec: PatternMotionSpec, *, mode: str, selected_block: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    direction_deg = float((selected_block % DIRS) * 45)
    with path.open("wb") as output:
        output.write(struct.pack("<ifi", 293390619, 1.0, 0))
        output.write(struct.pack("<b", 1))
        output.write(struct.pack("<iii", spec.width, spec.height, spec.frames_per_direction))
        for frame_index in range(spec.frames_per_direction):
            if mode == "plaid":
                output.write(make_plaid_frame(spec, direction_deg, frame_index, STIMULUS_CONTRAST))
            else:
                output.write(make_grating_frame(spec, direction_deg, frame_index, STIMULUS_CONTRAST))
    return path


def make_stimulus_frames(spec: PatternMotionSpec, *, mode: str, selected_block: int) -> list[bytes]:
    direction_deg = float((selected_block % DIRS) * 45)
    frames: list[bytes] = []
    for frame_index in range(spec.frames_per_direction):
        if mode == "plaid":
            frames.append(make_plaid_frame(spec, direction_deg, frame_index, STIMULUS_CONTRAST))
        else:
            frames.append(make_grating_frame(spec, direction_deg, frame_index, STIMULUS_CONTRAST))
    return frames


def make_input_conv_frames(
    spec: PatternMotionSpec,
    *,
    mode: str,
    selected_block: int,
    source_camera_index: int = 0,
) -> list[nb.InputConvFrame]:
    frames: list[nb.InputConvFrame] = []
    for frame_index, payload in enumerate(make_stimulus_frames(spec, mode=mode, selected_block=selected_block)):
        frames.append(
            nb.InputConvFrame(
                time_step=frame_index * spec.v1_update_interval_steps,
                source_camera_index=source_camera_index,
                width=spec.width,
                height=spec.height,
                channels=1,
                bytes=payload,
            )
        )
    return frames


def _lif_layer(count: int, dense_name: str) -> nb.NeuronLayer:
    return nb.NeuronLayer.lif_double(
        count,
        dense_name=dense_name,
        tau_m_ms=nb.float32(20.0),
        tau_exc_ms=nb.float32(5.0),
        tau_inh_ms=nb.float32(10.0),
        v_rest=nb.float32(-60.0),
        v_reset=nb.float32(-60.0),
        v_threshold=nb.float32(-50.0),
        Eexc=nb.float32(0.0),
        Einh=nb.float32(-80.0),
        R=nb.float32(1.0),
        t_ref=nb.float32(2.0),
        dense_steps_to_keep=64,
    )


def _append_connection(network: nb.Network, source: list[int], target: list[int], types: list[int], weights: list[float]) -> None:
    if not source:
        return
    network.connect(
        nb.Connection(
            source=source,
            target=target,
            synapse_type=types,
            weight=weights,
            max_weight=weights,
            delay=[1] * len(source),
        )
    )


def _sparse_connect(src: int, dst: int, stride: int) -> bool:
    return ((src * 1103515245 + dst * 12345 + 0x9E3779B9) & 0xFFFFFFFF) % stride == 0


def _circular_distance_deg(a: int, b: int) -> float:
    diff = abs(a - b)
    return float(min(diff, DIRS - diff) * 45)


def _pds_center_x(spec: PatternMotionSpec, block_x: int) -> float:
    return ((block_x + 0.5) * spec.width / spec.pds_grid_width) - 0.5


def _pds_center_y(spec: PatternMotionSpec, block_y: int) -> float:
    return ((block_y + 0.5) * spec.height / spec.pds_grid_height) - 0.5


def build_pattern_motion_network(
    spec: PatternMotionSpec,
    stimulus_file: Path | None,
    *,
    dynamic_input: bool = False,
    stimulus_mode: str = "grating",
) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(
        nb.NeuronLayer.poisson_rate(
            spec.v1_count,
            dense_name=spec.dense_name,
            rate_bias_hz=1.0,
            rate_gain_hz_per_current=1.0,
            dense_steps_to_keep=64,
        )
    )
    network.add_layer(_lif_layer(spec.cds_count, spec.dense_name))
    network.add_layer(_lif_layer(spec.pds_count, spec.dense_name))
    network.add_layer(_lif_layer(spec.pds_fs_count, spec.dense_name))
    network.add_layer(_lif_layer(spec.lip_count, spec.dense_name))

    spatial_count = spec.width * spec.height
    src: list[int] = []
    dst: list[int] = []
    typ: list[int] = []
    weight: list[float] = []

    def add(pre: int, post: int, synapse_type: int, synapse_weight: float) -> None:
        src.append(original_id(pre))
        dst.append(original_id(post))
        typ.append(int(synapse_type))
        weight.append(float(synapse_weight))
        if len(src) >= 50000:
            _append_connection(network, src.copy(), dst.copy(), typ.copy(), weight.copy())
            src.clear()
            dst.clear()
            typ.clear()
            weight.clear()

    for direction in range(DIRS):
        for spatial in range(spatial_count):
            add(v1_neuron_id(spec, direction, spatial), cds_neuron_id(spec, direction, spatial), 0, V1_CDS_EXC_WEIGHT)
            add(v1_neuron_id(spec, (direction + 4) % DIRS, spatial), cds_neuron_id(spec, direction, spatial), 1, V1_CDS_INH_WEIGHT)

    for spatial in range(spatial_count):
        src_x = spatial % spec.width
        src_y = spatial // spec.width
        for src_dir in range(DIRS):
            cds_id = cds_neuron_id(spec, src_dir, spatial)
            for dst_dir in range(DIRS):
                directional_gain = math.cos(2.0 * math.pi * (dst_dir - src_dir) / DIRS)
                if abs(directional_gain) < 1.0e-6:
                    continue
                for dst_local in range(spec.pds_pool_size):
                    bx = dst_local % spec.pds_grid_width
                    by = dst_local // spec.pds_grid_width
                    dx = _pds_center_x(spec, bx) - src_x
                    dy = _pds_center_y(spec, by) - src_y
                    spatial_gain = math.exp(-(dx * dx + dy * dy) / (2.0 * CDS_PDS_POOL_SIGMA_PIXELS * CDS_PDS_POOL_SIGMA_PIXELS))
                    signed_weight = CDS_PDS_ALPHA * directional_gain * spatial_gain
                    if abs(signed_weight) < CDS_PDS_WEIGHT_EPSILON:
                        continue
                    if signed_weight > 0.0:
                        add(cds_id, pds_neuron_id(spec, dst_dir, dst_local), 0, signed_weight)
                    else:
                        add(cds_id, pds_fs_neuron_id(spec, dst_dir, dst_local), 0, -signed_weight)

    for pds_fs in range(spec.pds_fs_count):
        dst_dir = pds_fs // spec.pds_pool_size
        dst_local = pds_fs % spec.pds_pool_size
        add(spec.pds_fs_base + pds_fs, pds_neuron_id(spec, dst_dir, dst_local), 1, PDS_FS_TO_PDS_INH_WEIGHT)

    for src_dir in range(DIRS):
        for src_local in range(spec.pds_pool_size):
            src_x = _pds_center_x(spec, src_local % spec.pds_grid_width)
            src_y = _pds_center_y(spec, src_local // spec.pds_grid_width)
            src_neuron = pds_neuron_id(spec, src_dir, src_local)
            for dst_dir in range(DIRS):
                dtheta = _circular_distance_deg(src_dir, dst_dir)
                directional_gain = math.exp(-(dtheta * dtheta) / (2.0 * PDS_TUNED_DIR_SIGMA_DEG * PDS_TUNED_DIR_SIGMA_DEG))
                for dst_local in range(spec.pds_pool_size):
                    dx = _pds_center_x(spec, dst_local % spec.pds_grid_width) - src_x
                    dy = _pds_center_y(spec, dst_local // spec.pds_grid_width) - src_y
                    spatial_gain = math.exp(-(dx * dx + dy * dy) / (2.0 * PDS_TUNED_LOC_SIGMA_PIXELS * PDS_TUNED_LOC_SIGMA_PIXELS))
                    tuned_weight = PDS_FS_EXC_WEIGHT * directional_gain * spatial_gain
                    if tuned_weight >= PDS_TUNED_WEIGHT_EPSILON:
                        add(src_neuron, pds_fs_neuron_id(spec, dst_dir, dst_local), 0, tuned_weight)

    for pds in range(spec.pds_count):
        src_dir = pds // spec.pds_pool_size
        src_neuron = spec.pds_base + pds
        for lip in range(spec.lip_pool_size):
            dst_neuron = lip_neuron_id(spec, src_dir, lip)
            if _sparse_connect(src_neuron, dst_neuron, 2):
                add(src_neuron, dst_neuron, 0, PDS_TO_LIP_WEIGHT)

    for src_dir in range(DIRS):
        for dst_dir in range(DIRS):
            if src_dir == dst_dir:
                continue
            delta = 2.0 * math.pi * (src_dir - dst_dir) / DIRS
            inh_strength = max(0.0, -math.cos(delta))
            if inh_strength <= 0.0:
                continue
            for src_idx in range(spec.lip_pool_size):
                for dst_idx in range(spec.lip_pool_size):
                    src_neuron = lip_neuron_id(spec, src_dir, src_idx)
                    dst_neuron = lip_neuron_id(spec, dst_dir, dst_idx)
                    if _sparse_connect(src_neuron, dst_neuron, 10):
                        add(src_neuron, dst_neuron, 1, LIP_INHIBITION_WEIGHT * inh_strength)

    _append_connection(network, src, dst, typ, weight)
    input_conv_kwargs = dict(
        width=spec.width,
        height=spec.height,
        channels=1,
        speed=V1_SPEED,
        update_timestep=spec.v1_update_interval_steps,
        output_target="dense_subnetwork",
        target_dense_subnetwork_name=spec.dense_name,
        output_source_indices=list(range(spec.v1_count)),
        output_target_neuron_ids=list(range(spec.v1_count)),
        output_pending_channel=2,
        output_scale=1.0,
        output_overwrite=False,
    )
    if dynamic_input:
        dynamic_kwargs = dict(
            input_conv_kwargs,
            input_frame_missing_policy="hold_last",
            input_frame_max_lag_steps=spec.v1_update_interval_steps,
        )
        if stimulus_mode == "plaid":
            network.add_input_conv(nb.InputConv.v1_plaid(**dynamic_kwargs))
        else:
            network.add_input_conv(nb.InputConv.v1_grating(**dynamic_kwargs))
    else:
        if stimulus_file is None:
            raise ValueError("stimulus_file is required unless dynamic_input=True")
        network.add_input_conv(
            nb.InputConv.v1_file(
                stimulus_file_path=str(stimulus_file),
                frame_hold_steps=1,
                **input_conv_kwargs,
            )
        )
    return network


def winner_dir(values: list[int] | list[float], neurons_per_dir: int) -> int:
    best_dir = 0
    best_value = -1.0e30
    for direction in range(DIRS):
        total = sum(values[direction * neurons_per_dir : (direction + 1) * neurons_per_dir])
        if total > best_value:
            best_value = float(total)
            best_dir = direction
    return best_dir


def count_layer_firings(output_ids: list[int], counts: list[int], base: int, count: int, neurons_per_dir: int, *, spatial_major_interleaved: bool = False) -> None:
    for neuron in output_ids:
        if base <= neuron < base + count:
            local = neuron - base
            mapped = local
            if spatial_major_interleaved:
                direction = local % DIRS
                spatial = local // DIRS
                mapped = direction * neurons_per_dir + spatial
            counts[mapped] += 1


def write_vector_tsv(path: Path, values: list[int] | list[float], neurons_per_dir: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dir_count = len(values) // neurons_per_dir
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("dir")
        for local in range(neurons_per_dir):
            output.write(f"\tn{local}")
        output.write("\n")
        for direction in range(dir_count):
            output.write(str(direction))
            for local in range(neurons_per_dir):
                output.write(f"\t{values[direction * neurons_per_dir + local]}")
            output.write("\n")


def _heat_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    red = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 3.0)))
    green = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 2.0)))
    blue = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 1.0)))
    return int(255.0 * red), int(255.0 * green), int(255.0 * blue)


def write_direction_montage_bmp(
    path: Path,
    values: list[int] | list[float],
    map_width: int,
    map_height: int,
    neurons_per_dir: int,
) -> None:
    if len(values) != DIRS * neurons_per_dir:
        raise ValueError("direction montage values must contain DIRS * neurons_per_dir entries")
    border = 2
    tile_cols = 4
    tile_rows = 2
    canvas_width = tile_cols * map_width + (tile_cols + 1) * border
    canvas_height = tile_rows * map_height + (tile_rows + 1) * border
    pixels = [(18, 18, 32)] * (canvas_width * canvas_height)
    min_value = float(min(values)) if values else 0.0
    max_value = float(max(values)) if values else 0.0
    denom = max(max_value - min_value, 1.0e-6)
    for direction in range(DIRS):
        tile_x = direction % tile_cols
        tile_y = direction // tile_cols
        x0 = border + tile_x * (map_width + border)
        y0 = border + tile_y * (map_height + border)
        for y in range(map_height):
            for x in range(map_width):
                local = y * map_width + x
                normalized = (float(values[direction * neurons_per_dir + local]) - min_value) / denom
                pixels[(y0 + y) * canvas_width + (x0 + x)] = _heat_color(normalized)

    row_stride = ((canvas_width * 3 + 3) // 4) * 4
    pixel_data_size = row_stride * canvas_height
    file_size = 54 + pixel_data_size
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(b"BM")
        output.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        output.write(struct.pack("<IIIHHIIIIII", 40, canvas_width, canvas_height, 1, 24, 0, pixel_data_size, 2835, 2835, 0, 0))
        padding = b"\x00" * (row_stride - canvas_width * 3)
        for y in range(canvas_height - 1, -1, -1):
            row = bytearray()
            for x in range(canvas_width):
                red, green, blue = pixels[y * canvas_width + x]
                row.extend((blue, green, red))
            output.write(row)
            output.write(padding)
