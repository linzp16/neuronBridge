"""Python interface for the NeuronBridge neural simulation runtime."""

from __future__ import annotations

import os
from pathlib import Path
import time

from .config import DebugMonitorConfig
from .communication import InputConvFrameClient, InputConvFramePublisher, InputConvFrameServer
from .frames import (
    InputConvFrame,
    InputConvFrameRequest,
    InputConvPixelFormat,
    pack_input_conv_frame,
    pack_input_conv_frame_request,
    unpack_input_conv_frame,
    unpack_input_conv_frame_request,
)
from .model import (
    Connection,
    FeedbackProductEncoding,
    FeedbackSingleEncoding,
    InputConv,
    LearningRule,
    NativeParameter,
    Network,
    NeuronLayer,
    OuterDynamic,
    OuterDynamicConnection,
    SimulationConfig,
    float32,
    float32_array3,
    float32_array4,
    float32_array5,
    float32_list,
    float64,
    float64_list,
    int32,
    int32_list,
)
from .results import DebugMonitorResult
from . import nbnet
from .nbnet import (
    NbnetBuildResult,
    NbnetDescriptionBuilder,
    NbnetFileInfo,
    NbnetLayerHandle,
    NbnetValidationReport,
    NbnetWriteOptions,
    StreamingBuildOptions,
)

try:
    from . import _core
except ImportError:  # pragma: no cover - used before the native module is built.
    _core = None

from . import catalog

__all__ = [
    "DebugMonitorConfig",
    "DebugMonitorResult",
    "Connection",
    "FeedbackProductEncoding",
    "FeedbackSingleEncoding",
    "InputConv",
    "InputConvFrame",
    "InputConvFrameClient",
    "InputConvFramePublisher",
    "InputConvFrameRequest",
    "InputConvFrameServer",
    "InputConvPixelFormat",
    "LearningRule",
    "NativeParameter",
    "NbnetBuildResult",
    "NbnetDescriptionBuilder",
    "NbnetFileInfo",
    "NbnetLayerHandle",
    "NbnetValidationReport",
    "NbnetWriteOptions",
    "Network",
    "NeuronLayer",
    "OuterDynamic",
    "OuterDynamicConnection",
    "SimulationConfig",
    "Simulation",
    "StreamingBuildOptions",
    "backend_info",
    "catalog",
    "nbnet",
    "float32",
    "float32_array3",
    "float32_array4",
    "float32_array5",
    "float32_list",
    "float64",
    "float64_list",
    "int32",
    "int32_list",
    "open_debug_monitor",
    "pack_input_conv_frame",
    "pack_input_conv_frame_request",
    "unpack_input_conv_frame",
    "unpack_input_conv_frame_request",
]

__version__ = "0.1.0a2"


def backend_info() -> dict:
    """Return native bridge build information when the extension is available."""
    if _core is None:
        return {
            "package": "neuronbridge",
            "backend": "NeuronBridge runtime",
            "binding": "pybind11",
            "native_extension_loaded": False,
            "api_stage": "python-api",
        }
    info = dict(_core.get_build_info())
    info["native_extension_loaded"] = True
    return info


class Simulation:
    """High-level facade for the native C++ Simulation lifecycle."""

    def __init__(
        self,
        network: Network | str | os.PathLike[str],
        config: SimulationConfig,
        *,
        build_options: StreamingBuildOptions | None = None,
    ):
        if _core is None:
            raise RuntimeError("neuronbridge native extension is not built")
        if not isinstance(config, SimulationConfig):
            raise TypeError("Simulation expects a neuronbridge.SimulationConfig")
        self.config = config
        build_started = time.perf_counter()
        if isinstance(network, Network):
            if build_options is not None:
                raise TypeError("build_options is only valid when Simulation receives an .nbnet path")
            self.network: Network | None = network
            self.network_file: Path | None = None
            self._native = _core.Simulation(network.to_native(), config.to_native())
            self._build_stats = {
                "mode": "in_memory",
                "neuron_count": network.neuron_count,
                "connection_block_count": len(network.connections),
            }
        elif isinstance(network, (str, os.PathLike)):
            options = build_options or StreamingBuildOptions()
            if not isinstance(options, StreamingBuildOptions):
                raise TypeError("build_options must be a StreamingBuildOptions")
            self.network = None
            self.network_file = Path(network).resolve()
            self._native = _core.Simulation(
                os.fspath(self.network_file),
                config.to_native(),
                options.to_native(),
            )
            self._build_stats = dict(self._native.build_stats)
        else:
            raise TypeError("Simulation expects a neuronbridge.Network or an .nbnet path")
        self._build_stats["total_build_seconds"] = time.perf_counter() - build_started
        self._debug_monitor_config: DebugMonitorConfig | None = None

    @property
    def build_stats(self) -> dict:
        """Return construction-path diagnostics for this simulation."""
        return dict(self._build_stats)

    @property
    def initialized(self) -> bool:
        return bool(self._native.initialized)

    @property
    def dense_subnetwork_count(self) -> int:
        return int(self._native.dense_subnetwork_count)

    @property
    def input_conv_count(self) -> int:
        return int(self._native.input_conv_count)

    def init(self) -> "Simulation":
        self._native.init()
        return self

    def run(self, steps: int | None = None) -> "Simulation":
        self._native.run(self.config.steps if steps is None else int(steps))
        return self

    def enable_realtime(
        self,
        *,
        slot_steps: int = 0,
        max_advance_seconds: float = 0.0,
        first_section: float = 0.25,
        second_section: float = 0.5,
        third_section: float = 0.75,
    ) -> "Simulation":
        """Temporarily deferred experimental API; not in the stable support scope.

        Enable native real-time pacing for subsequent ``run_realtime`` calls.

        ``slot_steps=0`` lets the native runtime use the configured communication
        interval.  The section values control how aggressively non-essential
        events are restricted when simulation time approaches wall time.
        """
        self._native.enable_realtime(
            int(slot_steps),
            float(max_advance_seconds),
            float(first_section),
            float(second_section),
            float(third_section),
        )
        return self

    def disable_realtime(self) -> "Simulation":
        """Temporarily deferred experimental API; disable native real-time pacing."""
        self._native.disable_realtime()
        return self

    def run_realtime(self, steps: int | None = None) -> "Simulation":
        """Temporarily deferred experimental API; run with native real-time pacing."""
        self._native.run_realtime(self.config.steps if steps is None else int(steps))
        return self

    def reset_bench_profiling(self) -> "Simulation":
        """Temporarily deferred experimental API; reset native profiling counters."""
        self._native.reset_bench_profiling()
        return self

    def bench_profiling_snapshot(self) -> dict:
        """Temporarily deferred experimental API; return native timing counters."""
        return dict(self._native.bench_profiling_snapshot())

    def realtime_skip_counters(self) -> dict:
        """Temporarily deferred experimental API; return realtime skip counters."""
        return dict(self._native.realtime_skip_counters())

    def reset_realtime_skip_counters(self) -> "Simulation":
        """Temporarily deferred experimental API; reset realtime skip counters."""
        self._native.reset_realtime_skip_counters()
        return self

    def realtime_restriction_counts(self) -> dict:
        """Temporarily deferred experimental API; return restriction-level counts."""
        return dict(self._native.realtime_restriction_counts())

    def reset_realtime_restriction_counts(self) -> "Simulation":
        """Temporarily deferred experimental API; reset restriction-level counts."""
        self._native.reset_realtime_restriction_counts()
        return self

    def reset(self) -> "Simulation":
        """Reset dynamic simulation state while preserving current weights."""
        self._native.reset()
        return self

    def add_external_spikes(self, times: list[int], neuron_ids: list[int]) -> "Simulation":
        self._native.add_external_spikes(times, neuron_ids)
        return self

    def add_external_currents(self, times: list[int], neuron_ids: list[int], currents: list[float]) -> "Simulation":
        self._native.add_external_currents(times, neuron_ids, currents)
        return self

    def add_zmq_async_input_output_spike_driver(
        self,
        *,
        subscribe_address: str = "127.0.0.1",
        publish_port: int = 5567,
        subscribe_port: int = 5566,
        publish_topic: str = "/snn/output_spikes",
        subscribe_topic: str = "/robot/input_spikes",
        communication_interval: int = 10,
    ) -> "Simulation":
        self._native.add_zmq_async_input_output_spike_driver(
            subscribe_address,
            int(publish_port),
            int(subscribe_port),
            publish_topic,
            subscribe_topic,
            int(communication_interval),
        )
        return self

    def add_zmq_input_output_spike_driver(
        self,
        *,
        server_address: str = "127.0.0.1",
        server_port: int = 5565,
        communication_interval: int = 10,
    ) -> "Simulation":
        """Add the blocking REQ/REP spike driver.

        The native simulation sends one request at each communication event
        and blocks until the external REP server returns its input spikes.
        """
        self._native.add_zmq_input_output_spike_driver(
            server_address,
            int(server_port),
            int(communication_interval),
        )
        return self

    def add_input_conv_frames(self, source_name: str, frames: list[InputConvFrame] | tuple[InputConvFrame, ...]) -> "Simulation":
        if not self.initialized:
            self.init()
        self._native.add_input_conv_frames(source_name, [frame.to_native() if isinstance(frame, InputConvFrame) else frame for frame in frames])
        return self

    def clear_input_conv_frame_queue(self, source_name: str) -> "Simulation":
        self._native.clear_input_conv_frame_queue(source_name)
        return self

    def bind_input_conv_frame_source(self, index_or_name: int | str, source_name: str, source_camera_index: int = 0) -> "Simulation":
        self._native.bind_input_conv_frame_source(index_or_name, source_name, int(source_camera_index))
        return self

    def has_input_conv_frame_source_binding(self, index: int = 0) -> bool:
        return bool(self._native.has_input_conv_frame_source_binding(int(index)))

    def add_zmq_input_conv_frame_source(
        self,
        source_name: str,
        *,
        address: str = "127.0.0.1",
        port: int,
        max_payload_bytes: int = 0,
    ) -> int:
        if not self.initialized:
            self.init()
        return int(self._native.add_zmq_input_conv_frame_source(source_name, address, int(port), int(max_payload_bytes)))

    def add_input_conv_frame_server_source(
        self,
        source_name: str,
        server: InputConvFrameServer,
        *,
        bind_to: int | str | None = None,
        source_camera_index: int = 0,
        max_payload_bytes: int = 0,
    ) -> int:
        server.wait_ready()
        source_index = self.add_zmq_input_conv_frame_source(
            source_name,
            address=server.bind_address,
            port=server.port,
            max_payload_bytes=max_payload_bytes,
        )
        if bind_to is not None:
            self.bind_input_conv_frame_source(bind_to, source_name, source_camera_index)
        return source_index

    def add_zmq_async_input_conv_frame_source(
        self,
        source_name: str,
        *,
        subscribe_address: str = "127.0.0.1",
        subscribe_port: int,
        topic: str = "inputconv_frame",
        max_payload_bytes: int = 0,
        max_buffered_frames_per_camera: int = 8,
    ) -> int:
        if not self.initialized:
            self.init()
        return int(
            self._native.add_zmq_async_input_conv_frame_source(
                source_name,
                subscribe_address,
                int(subscribe_port),
                topic,
                int(max_payload_bytes),
                int(max_buffered_frames_per_camera),
            )
        )

    def input_conv_frame_source_status(self, source_name_or_index: str | int) -> dict:
        return dict(self._native.input_conv_frame_source_status(source_name_or_index))

    def neuron_state(self, original_neuron_id: int) -> dict:
        return dict(self._native.neuron_state(int(original_neuron_id)))

    def neuron_states(self, original_neuron_ids: list[int]) -> list[dict]:
        return [dict(state) for state in self._native.neuron_states([int(value) for value in original_neuron_ids])]

    def output_spikes(self) -> list[dict]:
        return [dict(spike) for spike in self._native.output_spikes()]

    def enable_debug_monitor(self, config: DebugMonitorConfig) -> "Simulation":
        self._debug_monitor_config = config
        self._native.enable_debug_monitor(config.to_native())
        return self

    def disable_debug_monitor(self) -> "Simulation":
        self._native.disable_debug_monitor()
        return self

    def flush(self) -> "Simulation":
        self._native.flush_debug_monitor()
        return self

    def publish_output(self) -> "Simulation":
        self._native.publish_output()
        return self

    def result(self) -> DebugMonitorResult:
        if self._debug_monitor_config is None:
            raise RuntimeError("DebugMonitor is not enabled for this Simulation")
        return DebugMonitorResult(self._debug_monitor_config.output_dir)

    def get_connection_weight(self, original_connection_index: int) -> float:
        return float(self._native.get_connection_weight(int(original_connection_index)))

    def set_connection_weight(self, original_connection_index: int, weight: float) -> "Simulation":
        self._native.set_connection_weight(int(original_connection_index), float(weight))
        return self

    def save_weights(self, path) -> "Simulation":
        self._native.save_weights(str(path))
        return self

    def load_weights(self, path) -> "Simulation":
        self._native.load_weights(str(path))
        return self

    def dense_subnetwork_name(self, index: int) -> str:
        return self._native.dense_subnetwork_name(int(index))

    def find_dense_subnetwork(self, name: str) -> int:
        return int(self._native.find_dense_subnetwork(name))

    def dense_subnetwork_weights(self, index: int) -> list[float]:
        return list(self._native.dense_subnetwork_weights(int(index)))

    def dense_subnetwork_snapshot(self, index: int | str) -> dict:
        if isinstance(index, str):
            return dict(self._native.dense_subnetwork_snapshot_by_name(index))
        return dict(self._native.dense_subnetwork_snapshot(int(index)))

    def reset_dense_subnetwork(self, index: int | str) -> "Simulation":
        if isinstance(index, str):
            resolved = self.find_dense_subnetwork(index)
            if resolved < 0:
                raise KeyError(f"dense subnetwork was not found: {index}")
            index = resolved
        self._native.reset_dense_subnetwork(int(index))
        return self

    def set_dense_subnetwork_full_firing_export_enabled(self, index: int | str, enabled: bool = True) -> "Simulation":
        if isinstance(index, str):
            resolved = self.find_dense_subnetwork(index)
            if resolved < 0:
                raise KeyError(f"dense subnetwork was not found: {index}")
            index = resolved
        self._native.set_dense_subnetwork_full_firing_export_enabled(int(index), bool(enabled))
        return self

    def input_conv_output_count(self, index: int) -> int:
        return int(self._native.input_conv_output_count(int(index)))

    def input_conv_output(self, index: int = 0) -> list[float]:
        """Return the latest InputConv output/rate buffer as host floats."""
        return [float(value) for value in self._native.input_conv_output(int(index))]

    def input_conv_input(self, index: int = 0) -> list[float]:
        """Return the latest InputConv input frame when the model exposes it."""
        return [float(value) for value in self._native.input_conv_input(int(index))]

    def input_conv_rate_maps(
        self,
        index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        directions: int = 8,
    ) -> list[list[list[float]]]:
        """Return latest InputConv rates as `[direction][y][x]` maps."""
        values = self.input_conv_output(index)
        width, height = self._resolve_input_conv_map_shape(index, width, height)
        if directions <= 0:
            raise ValueError("directions must be positive")
        expected = width * height * directions
        if len(values) != expected:
            raise ValueError(f"InputConv output has {len(values)} values; expected {expected} for {width}x{height}x{directions}")
        maps: list[list[list[float]]] = []
        plane = width * height
        for direction in range(directions):
            base = direction * plane
            maps.append([values[base + y * width : base + (y + 1) * width] for y in range(height)])
        return maps

    def plot_input_conv_rate_heatmap(
        self,
        index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        directions: int = 8,
        ax=None,
        cmap: str = "magma",
        title_prefix: str = "dir",
    ):
        """Plot latest InputConv direction maps as a 4x2 montage."""
        import math

        if ax is not None:
            axes = [ax]
            rows = cols = 1
        else:
            import matplotlib.pyplot as plt

            cols = min(4, directions)
            rows = int(math.ceil(directions / cols))
            _, axes_grid = plt.subplots(rows, cols, squeeze=False)
            axes = [axes_grid[row][col] for row in range(rows) for col in range(cols)]
        maps = self.input_conv_rate_maps(index, width=width, height=height, directions=directions)
        vmax = max((max(row) for plane in maps for row in plane), default=0.0)
        vmin = min((min(row) for plane in maps for row in plane), default=0.0)
        for direction, plane in enumerate(maps):
            axes[direction].imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[direction].set_title(f"{title_prefix} {direction}")
            axes[direction].set_xticks([])
            axes[direction].set_yticks([])
        for extra in axes[directions:]:
            extra.axis("off")
        return axes[0] if ax is not None else axes

    def enable_input_conv_monitor(self, index_or_name: int | str) -> "Simulation":
        self._native.enable_input_conv_monitor(index_or_name)
        return self

    def disable_input_conv_monitor(self, index_or_name: int | str) -> "Simulation":
        self._native.disable_input_conv_monitor(index_or_name)
        return self

    def outer_dynamic_spike_counter_snapshot(self, name: str) -> dict:
        return dict(self._native.outer_dynamic_spike_counter_snapshot(name))

    def clear_outer_dynamic_spike_counter(self, name: str) -> "Simulation":
        self._native.clear_outer_dynamic_spike_counter(name)
        return self

    def clear_outer_dynamic_spike_counter_slot(self, name: str, slot_id: int) -> "Simulation":
        self._native.clear_outer_dynamic_spike_counter_slot(name, int(slot_id))
        return self

    def outer_dynamic_state(self) -> dict:
        """Return the latest joint state emitted by an OuterDynamic model."""
        return dict(self._native.outer_dynamic_state())

    def outer_dynamic_states(self) -> list[dict]:
        """Return the latest state reported by every OuterDynamic instance."""
        return [dict(state) for state in self._native.outer_dynamic_states()]

    def reset_outer_dynamic_state(self, name: str, q: list[float] | tuple[float, ...], qd: list[float] | tuple[float, ...]) -> "Simulation":
        self._native.reset_outer_dynamic_state(name, list(q), list(qd))
        return self

    def set_outer_dynamic_desired_state(
        self,
        name: str,
        q_des: list[float] | tuple[float, ...],
        qd_des: list[float] | tuple[float, ...],
    ) -> "Simulation":
        self._native.set_outer_dynamic_desired_state(name, list(q_des), list(qd_des))
        return self

    def _resolve_input_conv_map_shape(self, index: int, width: int | None, height: int | None) -> tuple[int, int]:
        if width is not None and height is not None:
            return int(width), int(height)
        if 0 <= index < len(self.network.input_convs):
            parameters = self.network.input_convs[index].to_dict()["parameters"]
            width = int(parameters.get("width", width or 0))
            height = int(parameters.get("height", height or 0))
        if width is None or height is None or int(width) <= 0 or int(height) <= 0:
            raise ValueError("width and height are required when they cannot be inferred from the InputConv description")
        return int(width), int(height)


def open_debug_monitor(path) -> DebugMonitorResult:
    """Open a DebugMonitor output directory."""
    return DebugMonitorResult(path)
