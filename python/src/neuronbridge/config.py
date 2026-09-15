"""Configuration objects shared by native and Python-facing APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    from . import _core
except ImportError:  # pragma: no cover - extension is optional while scaffolding.
    _core = None


@dataclass(slots=True)
class DebugMonitorConfig:
    enabled: bool = True
    output_dir: str | Path = "debug_monitor"
    sample_interval_steps: int = 1
    flush_interval_steps: int = 100
    record_spikes: bool = True
    record_state: bool = True
    record_weights: bool = False
    record_pending_channels: bool = True
    record_outer_dynamic_state: bool = True
    all_neurons: bool = False
    neuron_ids: list[int] = field(default_factory=list)
    dense_local_neuron_ids: dict[str, list[int]] = field(default_factory=dict)
    record_inputconv_outputs: bool = True
    record_inputconv_inputs: bool = False
    record_inputconv_internal_state: bool = False
    monitor_all_inputconv: bool = False
    monitored_inputconv_indices: list[int] = field(default_factory=list)
    monitored_inputconv_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "output_dir": str(self.output_dir),
            "sample_interval_steps": self.sample_interval_steps,
            "flush_interval_steps": self.flush_interval_steps,
            "record_spikes": self.record_spikes,
            "record_state": self.record_state,
            "record_weights": self.record_weights,
            "record_pending_channels": self.record_pending_channels,
            "record_outer_dynamic_state": self.record_outer_dynamic_state,
            "all_neurons": self.all_neurons,
            "neuron_ids": list(self.neuron_ids),
            "dense_local_neuron_ids": {
                str(name): [int(value) for value in values]
                for name, values in self.dense_local_neuron_ids.items()
            },
            "record_inputconv_outputs": self.record_inputconv_outputs,
            "record_inputconv_inputs": self.record_inputconv_inputs,
            "record_inputconv_internal_state": self.record_inputconv_internal_state,
            "monitor_all_inputconv": self.monitor_all_inputconv,
            "monitored_inputconv_indices": list(self.monitored_inputconv_indices),
            "monitored_inputconv_names": list(self.monitored_inputconv_names),
        }

    def to_native(self):
        """Create the native DebugMonitorConfig object if the bridge is built."""
        if _core is None:
            raise RuntimeError("neuronbridge native extension is not built")
        native = _core.DebugMonitorConfig()
        for key, value in self.to_dict().items():
            setattr(native, key, value)
        return native
