"""Python-friendly network description objects for neuronbridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from . import _core
except ImportError:  # pragma: no cover - extension is optional before build.
    _core = None


ParameterScalar = bool | int | float | str
ParameterList = list[bool] | list[int] | list[float] | list[str]


@dataclass(frozen=True, slots=True)
class NativeParameter:
    """Typed native parameter value for factories that require exact C++ types."""

    kind: str
    value: ParameterScalar | ParameterList

    def __post_init__(self) -> None:
        valid_kinds = {
            "int32",
            "float32",
            "float64",
            "int32_list",
            "float32_list",
            "float64_list",
            "float32_array3",
            "float32_array4",
        }
        if self.kind not in valid_kinds:
            raise ValueError(f"unsupported native parameter kind: {self.kind}")

    @property
    def __neuronbridge_param_kind__(self) -> str:
        return self.kind


ParameterValue = ParameterScalar | ParameterList | NativeParameter


def _require_native():
    if _core is None:
        raise RuntimeError("neuronbridge native extension is not built")
    return _core


def _validate_parameters(parameters: dict[str, Any]) -> dict[str, ParameterValue]:
    normalized: dict[str, ParameterValue] = {}
    for key, value in parameters.items():
        if not isinstance(key, str):
            raise TypeError("parameter names must be strings")
        if isinstance(value, NativeParameter):
            normalized[key] = value
            continue
        if isinstance(value, (bool, int, float, str)):
            normalized[key] = value
            continue
        if isinstance(value, list):
            if not value:
                normalized[key] = []
                continue
            first_type = bool if isinstance(value[0], bool) else type(value[0])
            if first_type not in (bool, int, float, str):
                raise TypeError(f"unsupported parameter list type for {key!r}")
            if not all(isinstance(item, first_type) for item in value):
                raise TypeError(f"mixed parameter list types are not supported for {key!r}")
            normalized[key] = value
            continue
        raise TypeError(f"unsupported parameter type for {key!r}: {type(value).__name__}")
    return normalized


def int32(value: int) -> NativeParameter:
    return NativeParameter("int32", int(value))


def float32(value: float) -> NativeParameter:
    return NativeParameter("float32", float(value))


def float64(value: float) -> NativeParameter:
    return NativeParameter("float64", float(value))


def int32_list(values: list[int]) -> NativeParameter:
    return NativeParameter("int32_list", [int(value) for value in values])


def float32_list(values: list[float]) -> NativeParameter:
    return NativeParameter("float32_list", [float(value) for value in values])


def float64_list(values: list[float]) -> NativeParameter:
    return NativeParameter("float64_list", [float(value) for value in values])


def float32_array3(values: list[float]) -> NativeParameter:
    if len(values) != 3:
        raise ValueError("float32_array3 requires exactly 3 values")
    return NativeParameter("float32_array3", [float(value) for value in values])


def float32_array4(values: list[float]) -> NativeParameter:
    if len(values) != 4:
        raise ValueError("float32_array4 requires exactly 4 values")
    return NativeParameter("float32_array4", [float(value) for value in values])


def _plain_parameter_value(value: ParameterValue):
    if isinstance(value, NativeParameter):
        return value.value
    return value


def _plain_parameters(parameters: dict[str, ParameterValue]) -> dict[str, Any]:
    return {key: _plain_parameter_value(value) for key, value in parameters.items()}


def _merge_parameters(base: dict[str, ParameterValue], extra: dict[str, ParameterValue]) -> dict[str, ParameterValue]:
    merged = dict(base)
    merged.update(extra)
    return merged


def _as_list(value: int | float | list[int] | list[float], *, name: str, cast):
    if isinstance(value, list):
        return [cast(item) for item in value]
    return [cast(value)]


@dataclass(slots=True)
class NeuronLayer:
    model: str
    count: int
    update_timestep: int = 1
    monitored: bool = False
    output: bool = False
    communication_input: bool = False
    parameters: dict[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("NeuronLayer.count must be positive")
        if self.update_timestep <= 0:
            raise ValueError("NeuronLayer.update_timestep must be positive")
        self.parameters = _validate_parameters(dict(self.parameters))

    @classmethod
    def input_spike(cls, count: int, **parameters: ParameterValue) -> "NeuronLayer":
        return cls("InputSpikeNeuronModel", count, parameters=dict(parameters))

    @classmethod
    def input_current(cls, count: int, **parameters: ParameterValue) -> "NeuronLayer":
        return cls("InputCurrentNeuronModel", count, parameters=dict(parameters))

    @classmethod
    def lif_decay(cls, count: int, **parameters: ParameterValue) -> "NeuronLayer":
        return cls("TimeDrivenLIF_Exponential_Decay", count, parameters=dict(parameters))

    @classmethod
    def lif_double(cls, count: int, *, dense_name: str | None = None, **parameters: ParameterValue) -> "NeuronLayer":
        params = dict(parameters)
        if dense_name is not None:
            params["dense_subnetwork_name"] = dense_name
        return cls("TimeDrivenLIF_Exponential_double", count, parameters=params)

    @classmethod
    def poisson_rate(
        cls,
        count: int,
        *,
        dense_name: str | None = None,
        rate_bias_hz: float | None = None,
        rate_gain_hz_per_current: float | None = None,
        **parameters: ParameterValue,
    ) -> "NeuronLayer":
        params = dict(parameters)
        if dense_name is not None:
            params["dense_subnetwork_name"] = dense_name
        if rate_bias_hz is not None:
            params["poisson_rate_bias_hz"] = float32(rate_bias_hz)
        if rate_gain_hz_per_current is not None:
            params["poisson_rate_gain_hz_per_current"] = float32(rate_gain_hz_per_current)
        return cls("PoissonRate", count, parameters=params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "count": self.count,
            "update_timestep": self.update_timestep,
            "monitored": self.monitored,
            "output": self.output,
            "communication_input": self.communication_input,
            "parameters": _plain_parameters(self.parameters),
        }


@dataclass(slots=True)
class Connection:
    source: int | list[int]
    target: int | list[int]
    synapse_type: int | list[int] = 0
    weight: float | list[float] = 1.0
    max_weight: float | list[float] | None = None
    delay: int | list[int] = 1
    synapse_rule: int | list[int] = -1
    trigger_rule: int | list[int] = -1

    def normalized(self) -> dict[str, list[int] | list[float]]:
        source = _as_list(self.source, name="source", cast=int)
        target = _as_list(self.target, name="target", cast=int)
        count = max(len(source), len(target))
        if len(source) == 1 and count > 1:
            source *= count
        if len(target) == 1 and count > 1:
            target *= count
        if len(source) != len(target):
            raise ValueError("Connection.source and target lengths must match or be scalar")

        def expand(value, cast, name):
            values = _as_list(value, name=name, cast=cast)
            if len(values) == 1 and count > 1:
                values *= count
            if len(values) != count:
                raise ValueError(f"Connection.{name} length must be 1 or {count}")
            return values

        weight = expand(self.weight, float, "weight")
        max_weight_source = self.weight if self.max_weight is None else self.max_weight
        return {
            "source": source,
            "target": target,
            "synapse_type": expand(self.synapse_type, int, "synapse_type"),
            "weight": weight,
            "max_weight": expand(max_weight_source, float, "max_weight"),
            "delay": expand(self.delay, int, "delay"),
            "synapse_rule": expand(self.synapse_rule, int, "synapse_rule"),
            "trigger_rule": expand(self.trigger_rule, int, "trigger_rule"),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.normalized()


@dataclass(slots=True)
class LearningRule:
    name: str
    parameters: dict[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parameters = _validate_parameters(dict(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parameters": _plain_parameters(self.parameters)}


@dataclass(slots=True)
class FeedbackProductEncoding:
    neuron_indices_by_joint: list[list[int]] = field(default_factory=list)
    bins: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.neuron_indices_by_joint = [[int(value) for value in joint] for joint in self.neuron_indices_by_joint]
        self.bins = [int(value) for value in self.bins]

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_indices_by_joint": [list(joint) for joint in self.neuron_indices_by_joint],
            "bins": list(self.bins),
        }


@dataclass(slots=True)
class FeedbackSingleEncoding:
    neuron_indices_by_joint_variable: list[list[list[int]]] = field(default_factory=list)
    bins: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.neuron_indices_by_joint_variable = [
            [[int(value) for value in variable] for variable in joint]
            for joint in self.neuron_indices_by_joint_variable
        ]
        self.bins = [int(value) for value in self.bins]

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_indices_by_joint_variable": [
                [list(variable) for variable in joint] for joint in self.neuron_indices_by_joint_variable
            ],
            "bins": list(self.bins),
        }


@dataclass(slots=True)
class OuterDynamic:
    model: str
    name: str = ""
    parameters: dict[str, ParameterValue] = field(default_factory=dict)
    update_timestep: int = 1
    communication_interval: int = 1
    queue_index: int = 0
    gc_neuron_indices_by_joint: list[list[int]] = field(default_factory=list)
    mf_neuron_indices_by_joint: list[list[int]] = field(default_factory=list)
    cf_positive_neuron_indices_by_joint: list[list[int]] = field(default_factory=list)
    cf_negative_neuron_indices_by_joint: list[list[int]] = field(default_factory=list)
    dcn_positive_neuron_indices_by_joint: list[list[int]] = field(default_factory=list)
    dcn_negative_neuron_indices_by_joint: list[list[int]] = field(default_factory=list)
    state_feedback_product: FeedbackProductEncoding = field(default_factory=FeedbackProductEncoding)
    state_feedback_single: FeedbackSingleEncoding = field(default_factory=FeedbackSingleEncoding)
    error_feedback_product: FeedbackProductEncoding = field(default_factory=FeedbackProductEncoding)
    error_feedback_single: FeedbackSingleEncoding = field(default_factory=FeedbackSingleEncoding)

    def __post_init__(self) -> None:
        if self.update_timestep <= 0:
            raise ValueError("OuterDynamic.update_timestep must be positive")
        if self.communication_interval <= 0:
            raise ValueError("OuterDynamic.communication_interval must be positive")
        if self.queue_index < 0:
            raise ValueError("OuterDynamic.queue_index must be non-negative")
        self.parameters = _validate_parameters(dict(self.parameters))
        for field_name in (
            "gc_neuron_indices_by_joint",
            "mf_neuron_indices_by_joint",
            "cf_positive_neuron_indices_by_joint",
            "cf_negative_neuron_indices_by_joint",
            "dcn_positive_neuron_indices_by_joint",
            "dcn_negative_neuron_indices_by_joint",
        ):
            setattr(self, field_name, [[int(value) for value in joint] for joint in getattr(self, field_name)])

    @classmethod
    def spike_counter(cls, *, name: str = "counter_sink", slot_count: int, type_count: int = 2) -> "OuterDynamic":
        return cls(
            "OuterDynamicSpikeCounter",
            name=name,
            parameters={"slot_count": int(slot_count), "type_count": int(type_count)},
        )

    @classmethod
    def planar_arm_2dof(cls, **kwargs: Any) -> "OuterDynamic":
        parameters = dict(kwargs.pop("parameters", {}))
        return cls._planar_arm(
            "PlanarArm2DOFPinocchio",
            parameters,
            kwargs,
        )

    @classmethod
    def strict_matlab_planar_arm_2dof(cls, **kwargs: Any) -> "OuterDynamic":
        parameters = dict(kwargs.pop("parameters", {}))
        return cls._planar_arm(
            "StrictMatlabPlanarArm2DOFOuterDynamic",
            parameters,
            kwargs,
        )

    @classmethod
    def _planar_arm(cls, model: str, parameters: dict[str, ParameterValue], kwargs: dict[str, Any]) -> "OuterDynamic":
        double_list_keys = (
            "link_lengths",
            "link_masses",
            "link_inertias",
            "joint_damping",
            "pd_kp",
            "pd_kd",
            "trajectory_amplitudes",
            "trajectory_phase_offsets",
            "angle_min_deg",
            "angle_max_deg",
            "velocity_min_deg_s",
            "velocity_max_deg_s",
            "velocity_range_deg_s",
            "torque_scale",
            "cf_angle_norm_deg",
            "cf_velocity_norm_deg_s",
        )
        double_keys = (
            "gravity",
            "trajectory_frequency_hz",
            "circle_center_x",
            "circle_center_y",
            "circle_radius",
            "dcn_torque_gain",
            "cf_mix_position",
            "spike_cf_max",
        )
        int_keys = (
            "spike_retention_steps",
            "state_feedback_delay_steps",
            "state_feedback_repeats_per_update",
            "state_feedback_repeat_period_steps",
            "error_feedback_delay_steps",
            "error_feedback_window_steps",
            "error_feedback_sample_count",
            "error_feedback_seed",
        )
        for key in double_list_keys:
            if key in kwargs:
                parameters[key] = float64_list(kwargs.pop(key))
        for key in double_keys:
            if key in kwargs:
                parameters[key] = float64(kwargs.pop(key))
        for key in int_keys:
            if key in kwargs:
                parameters[key] = int32(kwargs.pop(key))
        return cls(model, parameters=parameters, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "name": self.name,
            "parameters": _plain_parameters(self.parameters),
            "update_timestep": self.update_timestep,
            "communication_interval": self.communication_interval,
            "queue_index": self.queue_index,
            "gc_neuron_indices_by_joint": [list(joint) for joint in self.gc_neuron_indices_by_joint],
            "mf_neuron_indices_by_joint": [list(joint) for joint in self.mf_neuron_indices_by_joint],
            "cf_positive_neuron_indices_by_joint": [list(joint) for joint in self.cf_positive_neuron_indices_by_joint],
            "cf_negative_neuron_indices_by_joint": [list(joint) for joint in self.cf_negative_neuron_indices_by_joint],
            "dcn_positive_neuron_indices_by_joint": [list(joint) for joint in self.dcn_positive_neuron_indices_by_joint],
            "dcn_negative_neuron_indices_by_joint": [list(joint) for joint in self.dcn_negative_neuron_indices_by_joint],
            "state_feedback_product": self.state_feedback_product.to_dict(),
            "state_feedback_single": self.state_feedback_single.to_dict(),
            "error_feedback_product": self.error_feedback_product.to_dict(),
            "error_feedback_single": self.error_feedback_single.to_dict(),
        }


@dataclass(slots=True)
class OuterDynamicConnection:
    source: int | list[int]
    target_outer_dynamic: int | list[int]
    target_joint: int | list[int]
    synapse_type: int | list[int] = 0
    weight: float | list[float] = 1.0
    delay: int | list[int] = 0

    def normalized(self) -> dict[str, list[int] | list[float]]:
        source = _as_list(self.source, name="source", cast=int)
        target_outer_dynamic = _as_list(self.target_outer_dynamic, name="target_outer_dynamic", cast=int)
        target_joint = _as_list(self.target_joint, name="target_joint", cast=int)
        count = max(len(source), len(target_outer_dynamic), len(target_joint))

        def expand(value, cast, name):
            values = _as_list(value, name=name, cast=cast)
            if len(values) == 1 and count > 1:
                values *= count
            if len(values) != count:
                raise ValueError(f"OuterDynamicConnection.{name} length must be 1 or {count}")
            return values

        return {
            "source": expand(self.source, int, "source"),
            "target_outer_dynamic": expand(self.target_outer_dynamic, int, "target_outer_dynamic"),
            "target_joint": expand(self.target_joint, int, "target_joint"),
            "synapse_type": expand(self.synapse_type, int, "synapse_type"),
            "weight": expand(self.weight, float, "weight"),
            "delay": expand(self.delay, int, "delay"),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.normalized()


@dataclass(slots=True)
class InputConv:
    model: str = "InputConvV1"
    parameters: dict[str, ParameterValue] = field(default_factory=dict)
    update_timestep: int = 1
    queue_index: int = 0
    output_target: str = "main_network"
    target_dense_subnetwork_name: str = ""
    output_source_indices: list[int] = field(default_factory=list)
    output_target_neuron_ids: list[int] = field(default_factory=list)
    output_pending_channel: int = 0
    output_scale: float = 1.0
    output_scales: list[float] = field(default_factory=list)
    output_overwrite: bool = True

    def __post_init__(self) -> None:
        if self.update_timestep <= 0:
            raise ValueError("InputConv.update_timestep must be positive")
        if self.queue_index < 0:
            raise ValueError("InputConv.queue_index must be non-negative")
        if self.output_target not in {"main_network", "dense_subnetwork"}:
            raise ValueError("InputConv.output_target must be 'main_network' or 'dense_subnetwork'")
        if self.output_target == "dense_subnetwork" and not self.target_dense_subnetwork_name:
            raise ValueError("dense_subnetwork output requires target_dense_subnetwork_name")
        if len(self.output_source_indices) != len(self.output_target_neuron_ids):
            raise ValueError("InputConv output source and target id lists must have the same length")
        if self.output_scales and len(self.output_scales) != len(self.output_source_indices):
            raise ValueError("InputConv.output_scales must be empty or match output_source_indices length")
        self.parameters = _validate_parameters(dict(self.parameters))
        self.output_source_indices = [int(value) for value in self.output_source_indices]
        self.output_target_neuron_ids = [int(value) for value in self.output_target_neuron_ids]
        self.output_scales = [float(value) for value in self.output_scales]

    @classmethod
    def v1_bar(
        cls,
        *,
        width: int = 8,
        height: int = 8,
        channels: int = 1,
        speed: float = 1.5,
        bar_width: int = 2,
        motion_period_steps: int = 32,
        input_frame_missing_policy: str | None = None,
        input_frame_max_lag_steps: int | None = None,
        **kwargs: Any,
    ) -> "InputConv":
        parameters = dict(kwargs.pop("parameters", {}))
        dynamic_parameters = {}
        if input_frame_missing_policy is not None:
            dynamic_parameters["input_frame_missing_policy"] = str(input_frame_missing_policy)
        if input_frame_max_lag_steps is not None:
            dynamic_parameters["input_frame_max_lag_steps"] = int32(input_frame_max_lag_steps)
        parameters = _merge_parameters(
            {
                "width": int32(width),
                "height": int32(height),
                "channels": int32(channels),
                "stimulus_mode": "bar",
                "speed": float32(speed),
                "bar_width": int32(bar_width),
                "motion_period_steps": int32(motion_period_steps),
                **dynamic_parameters,
            },
            parameters,
        )
        return cls(parameters=parameters, **kwargs)

    @classmethod
    def v1_grating(
        cls,
        *,
        width: int = 8,
        height: int = 8,
        channels: int = 1,
        speed: float = 1.5,
        stimulus_temporal_speed: float = 0.08,
        grating_direction_deg: float = 0.0,
        spatial_frequency: float = 2.2,
        input_frame_missing_policy: str | None = None,
        input_frame_max_lag_steps: int | None = None,
        **kwargs: Any,
    ) -> "InputConv":
        parameters = dict(kwargs.pop("parameters", {}))
        dynamic_parameters = {}
        if input_frame_missing_policy is not None:
            dynamic_parameters["input_frame_missing_policy"] = str(input_frame_missing_policy)
        if input_frame_max_lag_steps is not None:
            dynamic_parameters["input_frame_max_lag_steps"] = int32(input_frame_max_lag_steps)
        parameters = _merge_parameters(
            {
                "width": int32(width),
                "height": int32(height),
                "channels": int32(channels),
                "stimulus_mode": "grating",
                "speed": float32(speed),
                "stimulus_temporal_speed": float32(stimulus_temporal_speed),
                "grating_direction_deg": float32(grating_direction_deg),
                "spatial_frequency": float32(spatial_frequency),
                **dynamic_parameters,
            },
            parameters,
        )
        return cls(parameters=parameters, **kwargs)

    @classmethod
    def v1_plaid(
        cls,
        *,
        width: int = 8,
        height: int = 8,
        channels: int = 1,
        speed: float = 1.5,
        stimulus_temporal_speed: float = 0.08,
        plaid_direction_a_deg: float = 0.0,
        plaid_direction_b_deg: float = 120.0,
        spatial_frequency: float = 2.2,
        input_frame_missing_policy: str | None = None,
        input_frame_max_lag_steps: int | None = None,
        **kwargs: Any,
    ) -> "InputConv":
        parameters = dict(kwargs.pop("parameters", {}))
        dynamic_parameters = {}
        if input_frame_missing_policy is not None:
            dynamic_parameters["input_frame_missing_policy"] = str(input_frame_missing_policy)
        if input_frame_max_lag_steps is not None:
            dynamic_parameters["input_frame_max_lag_steps"] = int32(input_frame_max_lag_steps)
        parameters = _merge_parameters(
            {
                "width": int32(width),
                "height": int32(height),
                "channels": int32(channels),
                "stimulus_mode": "plaid",
                "speed": float32(speed),
                "stimulus_temporal_speed": float32(stimulus_temporal_speed),
                "plaid_direction_a_deg": float32(plaid_direction_a_deg),
                "plaid_direction_b_deg": float32(plaid_direction_b_deg),
                "spatial_frequency": float32(spatial_frequency),
                **dynamic_parameters,
            },
            parameters,
        )
        return cls(parameters=parameters, **kwargs)

    @classmethod
    def v1_file(
        cls,
        *,
        stimulus_file_path: str,
        width: int,
        height: int,
        channels: int = 1,
        speed: float = 1.5,
        frame_hold_steps: int = 1,
        input_frame_missing_policy: str | None = None,
        input_frame_max_lag_steps: int | None = None,
        **kwargs: Any,
    ) -> "InputConv":
        parameters = dict(kwargs.pop("parameters", {}))
        dynamic_parameters = {}
        if input_frame_missing_policy is not None:
            dynamic_parameters["input_frame_missing_policy"] = str(input_frame_missing_policy)
        if input_frame_max_lag_steps is not None:
            dynamic_parameters["input_frame_max_lag_steps"] = int32(input_frame_max_lag_steps)
        parameters = _merge_parameters(
            {
                "width": int32(width),
                "height": int32(height),
                "channels": int32(channels),
                "stimulus_mode": "file",
                "stimulus_file_path": stimulus_file_path,
                "speed": float32(speed),
                "frame_hold_steps": int32(frame_hold_steps),
                **dynamic_parameters,
            },
            parameters,
        )
        return cls(parameters=parameters, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "parameters": _plain_parameters(self.parameters),
            "update_timestep": self.update_timestep,
            "queue_index": self.queue_index,
            "output_target": self.output_target,
            "target_dense_subnetwork_name": self.target_dense_subnetwork_name,
            "output_source_indices": list(self.output_source_indices),
            "output_target_neuron_ids": list(self.output_target_neuron_ids),
            "output_pending_channel": self.output_pending_channel,
            "output_scale": self.output_scale,
            "output_scales": list(self.output_scales),
            "output_overwrite": self.output_overwrite,
        }


@dataclass(slots=True)
class SimulationConfig:
    steps: int
    timestep: float
    queues: int = 1
    event_queue: str = "heap"
    timing_wheel_size: int = 0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("SimulationConfig.steps must be positive")
        if self.timestep <= 0:
            raise ValueError("SimulationConfig.timestep must be positive")
        if self.queues <= 0:
            raise ValueError("SimulationConfig.queues must be positive")
        if self.event_queue not in {"heap", "timing_wheel"}:
            raise ValueError("SimulationConfig.event_queue must be 'heap' or 'timing_wheel'")

    def to_native(self):
        core = _require_native()
        return core.SimulationConfig(
            self.steps,
            self.timestep,
            self.queues,
            self.event_queue,
            self.timing_wheel_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "timestep": self.timestep,
            "queues": self.queues,
            "event_queue": self.event_queue,
            "timing_wheel_size": self.timing_wheel_size,
        }


@dataclass(slots=True)
class Network:
    layers: list[NeuronLayer] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    learning_rules: list[LearningRule] = field(default_factory=list)
    outer_dynamics: list[OuterDynamic] = field(default_factory=list)
    outer_dynamic_connections: list[OuterDynamicConnection] = field(default_factory=list)
    input_convs: list[InputConv] = field(default_factory=list)

    def add_layer(self, layer: NeuronLayer) -> "Network":
        if not isinstance(layer, NeuronLayer):
            raise TypeError("Network.add_layer expects a NeuronLayer")
        self.layers.append(layer)
        return self

    def connect(self, connection: Connection) -> "Network":
        if not isinstance(connection, Connection):
            raise TypeError("Network.connect expects a Connection")
        self.connections.append(connection)
        return self

    def add_learning_rule(self, rule: LearningRule) -> "Network":
        if not isinstance(rule, LearningRule):
            raise TypeError("Network.add_learning_rule expects a LearningRule")
        self.learning_rules.append(rule)
        return self

    def add_outer_dynamic(self, outer_dynamic: OuterDynamic) -> "Network":
        if not isinstance(outer_dynamic, OuterDynamic):
            raise TypeError("Network.add_outer_dynamic expects an OuterDynamic")
        self.outer_dynamics.append(outer_dynamic)
        return self

    def connect_outer_dynamic(self, connection: OuterDynamicConnection) -> "Network":
        if not isinstance(connection, OuterDynamicConnection):
            raise TypeError("Network.connect_outer_dynamic expects an OuterDynamicConnection")
        self.outer_dynamic_connections.append(connection)
        return self

    def add_input_conv(self, input_conv: InputConv) -> "Network":
        if not isinstance(input_conv, InputConv):
            raise TypeError("Network.add_input_conv expects an InputConv")
        self.input_convs.append(input_conv)
        return self

    @property
    def neuron_count(self) -> int:
        return sum(layer.count for layer in self.layers)

    def to_native(self):
        core = _require_native()
        native = core.NetworkDescription()
        for layer in self.layers:
            native.add_layer(
                layer.model,
                layer.count,
                layer.update_timestep,
                layer.monitored,
                layer.output,
                layer.communication_input,
                layer.parameters,
            )
        for rule in self.learning_rules:
            native.add_learning_rule(rule.name, rule.parameters)
        for connection in self.connections:
            data = connection.normalized()
            native.add_connection(
                data["source"],
                data["target"],
                data["synapse_type"],
                data["weight"],
                data["max_weight"],
                data["delay"],
                data["synapse_rule"],
                data["trigger_rule"],
            )
        for outer_dynamic in self.outer_dynamics:
            native.add_outer_dynamic(
                outer_dynamic.model,
                outer_dynamic.name,
                outer_dynamic.parameters,
                outer_dynamic.update_timestep,
                outer_dynamic.communication_interval,
                outer_dynamic.queue_index,
                outer_dynamic.gc_neuron_indices_by_joint,
                outer_dynamic.mf_neuron_indices_by_joint,
                outer_dynamic.cf_positive_neuron_indices_by_joint,
                outer_dynamic.cf_negative_neuron_indices_by_joint,
                outer_dynamic.dcn_positive_neuron_indices_by_joint,
                outer_dynamic.dcn_negative_neuron_indices_by_joint,
                outer_dynamic.state_feedback_product.to_dict(),
                outer_dynamic.state_feedback_single.to_dict(),
                outer_dynamic.error_feedback_product.to_dict(),
                outer_dynamic.error_feedback_single.to_dict(),
            )
        for connection in self.outer_dynamic_connections:
            data = connection.normalized()
            native.add_outer_dynamic_connection(
                data["source"],
                data["target_outer_dynamic"],
                data["target_joint"],
                data["synapse_type"],
                data["weight"],
                data["delay"],
            )
        for input_conv in self.input_convs:
            native.add_input_conv(
                input_conv.model,
                input_conv.parameters,
                input_conv.update_timestep,
                input_conv.queue_index,
                input_conv.output_target,
                input_conv.target_dense_subnetwork_name,
                input_conv.output_source_indices,
                input_conv.output_target_neuron_ids,
                input_conv.output_pending_channel,
                input_conv.output_scale,
                input_conv.output_scales,
                input_conv.output_overwrite,
            )
        return native

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "connections": [connection.to_dict() for connection in self.connections],
            "learning_rules": [rule.to_dict() for rule in self.learning_rules],
            "outer_dynamics": [outer_dynamic.to_dict() for outer_dynamic in self.outer_dynamics],
            "outer_dynamic_connections": [connection.to_dict() for connection in self.outer_dynamic_connections],
            "input_convs": [input_conv.to_dict() for input_conv in self.input_convs],
        }
