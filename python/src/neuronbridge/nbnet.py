"""Low-memory authoring and loading for NeuronBridge ``.nbnet`` files.

The static network metadata is stored as JSON and the potentially large
connection table is stored as fixed-width binary records.  Connection records
are written to a temporary spool and copied in bounded chunks during finalize,
so callers do not need to keep the complete graph in Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .model import (
    Connection,
    InputConv,
    LearningRule,
    NativeParameter,
    Network,
    NeuronLayer,
    OuterDynamic,
    OuterDynamicConnection,
)


_MAGIC = b"NBNET01\0"
_VERSION_MAJOR = 1
_VERSION_MINOR = 0
_HEADER = struct.Struct("<8sHHIIQQ32s")
_CONNECTION = struct.Struct("<IIiffIii")
_FLAG_SHA256 = 1
_PARAMETER_MARKER = "__neuronbridge_parameter__"
_MAX_METADATA_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class StreamingBuildOptions:
    """Controls bounded-memory loading of a ``.nbnet`` description."""

    memory_budget_mb: int = 128
    mmap: bool = True
    verify_checksum: bool = True

    def __post_init__(self) -> None:
        if self.memory_budget_mb <= 0:
            raise ValueError("memory_budget_mb must be positive")

    def to_native(self):
        from . import _core

        if _core is None:
            raise RuntimeError("neuronbridge native extension is not built")
        return _core.StreamingBuildOptions(
            int(self.memory_budget_mb) * 1024 * 1024,
            bool(self.mmap),
            bool(self.verify_checksum),
        )


@dataclass(frozen=True, slots=True)
class NbnetWriteOptions:
    overwrite: bool = False
    fsync: bool = True


@dataclass(frozen=True, slots=True)
class NbnetLayerHandle:
    index: int
    first_neuron_id: int
    count: int


@dataclass(frozen=True, slots=True)
class NbnetBuildResult:
    path: Path
    file_bytes: int
    neuron_count: int
    connection_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class NbnetFileInfo:
    path: Path
    version: tuple[int, int]
    file_bytes: int
    metadata_bytes: int
    neuron_count: int
    connection_count: int
    checksum: str


@dataclass(frozen=True, slots=True)
class NbnetValidationReport:
    valid: bool
    info: NbnetFileInfo | None
    errors: tuple[str, ...]


def _encode_parameter(value: Any) -> Any:
    if isinstance(value, NativeParameter):
        return {
            _PARAMETER_MARKER: value.kind,
            "value": value.value,
        }
    if isinstance(value, dict):
        return {str(key): _encode_parameter(item) for key, item in value.items()}
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    elif isinstance(value, str):
        kind = "str"
    elif isinstance(value, (list, tuple)):
        if not value:
            kind = "int_list"
        elif isinstance(value[0], bool):
            kind = "bool_list"
        elif isinstance(value[0], int):
            kind = "int_list"
        elif isinstance(value[0], float):
            kind = "float_list"
        elif isinstance(value[0], str):
            kind = "str_list"
        else:
            raise TypeError(f"unsupported nbnet parameter list item: {type(value[0]).__name__}")
    else:
        raise TypeError(f"unsupported nbnet parameter value: {type(value).__name__}")
    return {_PARAMETER_MARKER: kind, "value": value}


def _object_metadata(value: Any) -> dict[str, Any]:
    result = value.to_dict()
    if hasattr(value, "parameters"):
        result["parameters"] = _encode_parameter(value.parameters)
    return result


def _sequence_length(value: Any) -> int | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _broadcast_get(value: Any, index: int, count: int, name: str) -> Any:
    length = _sequence_length(value)
    if length is None:
        return value
    if length == count:
        return value[index]
    if length == 1:
        return value[0]
    raise ValueError(f"{name} length must be 1 or {count}, got {length}")


class NbnetDescriptionBuilder:
    """Incrementally author a static ``.nbnet`` network description."""

    def __init__(self, path: str | os.PathLike[str], *, options: NbnetWriteOptions | None = None):
        self.path = Path(path)
        self.options = options or NbnetWriteOptions()
        self._metadata: dict[str, list[dict[str, Any]]] = {
            "layers": [],
            "learning_rules": [],
            "outer_dynamics": [],
            "outer_dynamic_connections": [],
            "input_convs": [],
        }
        self._neuron_count = 0
        self._connection_count = 0
        self._result: NbnetBuildResult | None = None
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        spool = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{self.path.name}.",
            suffix=".records.tmp",
            dir=self.path.parent,
            delete=False,
        )
        self._spool = spool
        self._spool_path = Path(spool.name)

    @property
    def result(self) -> NbnetBuildResult | None:
        return self._result

    def add_layer(self, layer: NeuronLayer) -> NbnetLayerHandle:
        self._ensure_open()
        if not isinstance(layer, NeuronLayer):
            raise TypeError("add_layer expects a NeuronLayer")
        if self._connection_count:
            raise RuntimeError("all layers must be added before the first connection batch")
        handle = NbnetLayerHandle(len(self._metadata["layers"]), self._neuron_count, layer.count)
        self._metadata["layers"].append(_object_metadata(layer))
        self._neuron_count += layer.count
        return handle

    def add_learning_rule(self, rule: LearningRule) -> int:
        self._ensure_open()
        if not isinstance(rule, LearningRule):
            raise TypeError("add_learning_rule expects a LearningRule")
        self._metadata["learning_rules"].append(_object_metadata(rule))
        return len(self._metadata["learning_rules"]) - 1

    def add_outer_dynamic(self, outer_dynamic: OuterDynamic) -> int:
        self._ensure_open()
        if not isinstance(outer_dynamic, OuterDynamic):
            raise TypeError("add_outer_dynamic expects an OuterDynamic")
        self._metadata["outer_dynamics"].append(_object_metadata(outer_dynamic))
        return len(self._metadata["outer_dynamics"]) - 1

    def connect_outer_dynamic(self, connection: OuterDynamicConnection) -> None:
        self._ensure_open()
        if not isinstance(connection, OuterDynamicConnection):
            raise TypeError("connect_outer_dynamic expects an OuterDynamicConnection")
        self._metadata["outer_dynamic_connections"].append(connection.to_dict())

    def add_input_conv(self, input_conv: InputConv) -> int:
        self._ensure_open()
        if not isinstance(input_conv, InputConv):
            raise TypeError("add_input_conv expects an InputConv")
        self._metadata["input_convs"].append(_object_metadata(input_conv))
        return len(self._metadata["input_convs"]) - 1

    def append_connections(
        self,
        source: Sequence[int],
        target: Sequence[int],
        *,
        synapse_type: int | Sequence[int] = 0,
        weight: float | Sequence[float] = 1.0,
        max_weight: float | Sequence[float] | None = None,
        delay: int | Sequence[int] = 1,
        synapse_rule: int | Sequence[int] = -1,
        trigger_rule: int | Sequence[int] = -1,
    ) -> int:
        """Append a batch without retaining it in the builder."""

        self._ensure_open()
        if not self._metadata["layers"]:
            raise RuntimeError("at least one layer must be added before connections")
        count = len(source)
        if len(target) != count:
            raise ValueError("source and target lengths must match")
        max_weight_values = weight if max_weight is None else max_weight
        fields = (
            (synapse_type, "synapse_type"),
            (weight, "weight"),
            (max_weight_values, "max_weight"),
            (delay, "delay"),
            (synapse_rule, "synapse_rule"),
            (trigger_rule, "trigger_rule"),
        )
        for values, name in fields:
            length = _sequence_length(values)
            if length not in (None, 1, count):
                raise ValueError(f"{name} length must be 1 or {count}, got {length}")

        for index in range(count):
            src = int(source[index])
            dst = int(target[index])
            syn_type = int(_broadcast_get(synapse_type, index, count, "synapse_type"))
            item_weight = float(_broadcast_get(weight, index, count, "weight"))
            item_max_weight = float(_broadcast_get(max_weight_values, index, count, "max_weight"))
            item_delay = int(_broadcast_get(delay, index, count, "delay"))
            item_rule = int(_broadcast_get(synapse_rule, index, count, "synapse_rule"))
            item_trigger = int(_broadcast_get(trigger_rule, index, count, "trigger_rule"))
            if src < 0 or dst < 0 or item_delay < 0:
                raise ValueError("source, target, and delay must be non-negative")
            if src >= self._neuron_count or dst >= self._neuron_count:
                raise ValueError(
                    f"connection neuron id is out of range [0, {self._neuron_count}): {src} -> {dst}"
                )
            if item_delay > 0xFFFFFFFF:
                raise ValueError("delay exceeds the nbnet v1 uint32 range")
            if not -(2**31) <= syn_type < 2**31:
                raise ValueError("synapse_type exceeds the nbnet v1 int32 range")
            if not -(2**31) <= item_rule < 2**31 or not -(2**31) <= item_trigger < 2**31:
                raise ValueError("learning-rule index exceeds the nbnet v1 int32 range")
            if not math.isfinite(item_weight) or not math.isfinite(item_max_weight):
                raise ValueError("weight and max_weight must be finite")
            self._spool.write(
                _CONNECTION.pack(
                    src,
                    dst,
                    syn_type,
                    item_weight,
                    item_max_weight,
                    item_delay,
                    item_rule,
                    item_trigger,
                )
            )
        self._connection_count += count
        return count

    def append_connection(self, connection: Connection) -> int:
        if not isinstance(connection, Connection):
            raise TypeError("append_connection expects a Connection")
        data = connection.normalized()
        return self.append_connections(**data)

    def finalize(self) -> NbnetBuildResult:
        if self._result is not None:
            return self._result
        self._ensure_open()
        if self.path.exists() and not self.options.overwrite:
            raise FileExistsError(f"refusing to overwrite existing nbnet file: {self.path}")

        metadata = json.dumps(
            self._metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(metadata)
        self._spool.flush()
        self._spool.seek(0)
        final_tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with final_tmp.open("w+b") as output:
                output.write(b"\0" * _HEADER.size)
                output.write(metadata)
                while chunk := self._spool.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                checksum = digest.digest()
                output.seek(0)
                output.write(
                    _HEADER.pack(
                        _MAGIC,
                        _VERSION_MAJOR,
                        _VERSION_MINOR,
                        _FLAG_SHA256,
                        _CONNECTION.size,
                        len(metadata),
                        self._connection_count,
                        checksum,
                    )
                )
                output.flush()
                if self.options.fsync:
                    os.fsync(output.fileno())
            os.replace(final_tmp, self.path)
        finally:
            if final_tmp.exists():
                final_tmp.unlink()
            self._close_spool()

        self._result = NbnetBuildResult(
            path=self.path.resolve(),
            file_bytes=self.path.stat().st_size,
            neuron_count=self._neuron_count,
            connection_count=self._connection_count,
            sha256=digest.hexdigest(),
        )
        return self._result

    def abort(self) -> None:
        self._close_spool()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("nbnet builder is closed")

    def _close_spool(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._spool.close()
        if self._spool_path.exists():
            self._spool_path.unlink()

    def __enter__(self) -> "NbnetDescriptionBuilder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None and self._result is None:
            self.finalize()
        else:
            self._close_spool()


def from_network(
    network: Network,
    path: str | os.PathLike[str],
    *,
    options: NbnetWriteOptions | None = None,
) -> NbnetBuildResult:
    """Convert an existing in-memory ``Network`` to ``.nbnet``."""

    if not isinstance(network, Network):
        raise TypeError("from_network expects a neuronbridge.Network")
    builder = NbnetDescriptionBuilder(path, options=options)
    try:
        for layer in network.layers:
            builder.add_layer(layer)
        for rule in network.learning_rules:
            builder.add_learning_rule(rule)
        for outer_dynamic in network.outer_dynamics:
            builder.add_outer_dynamic(outer_dynamic)
        for connection in network.outer_dynamic_connections:
            builder.connect_outer_dynamic(connection)
        for input_conv in network.input_convs:
            builder.add_input_conv(input_conv)
        for connection in network.connections:
            builder.append_connection(connection)
        return builder.finalize()
    except BaseException:
        builder.abort()
        raise


def from_batches(
    path: str | os.PathLike[str],
    layers: Iterable[NeuronLayer],
    connection_batches: Iterable[Mapping[str, Any]],
    *,
    learning_rules: Iterable[LearningRule] = (),
    outer_dynamics: Iterable[OuterDynamic] = (),
    outer_dynamic_connections: Iterable[OuterDynamicConnection] = (),
    input_convs: Iterable[InputConv] = (),
    options: NbnetWriteOptions | None = None,
) -> NbnetBuildResult:
    """Create ``.nbnet`` directly from iterable connection batches."""

    builder = NbnetDescriptionBuilder(path, options=options)
    try:
        for layer in layers:
            builder.add_layer(layer)
        for rule in learning_rules:
            builder.add_learning_rule(rule)
        for outer_dynamic in outer_dynamics:
            builder.add_outer_dynamic(outer_dynamic)
        for connection in outer_dynamic_connections:
            builder.connect_outer_dynamic(connection)
        for input_conv in input_convs:
            builder.add_input_conv(input_conv)
        for batch in connection_batches:
            builder.append_connections(**dict(batch))
        return builder.finalize()
    except BaseException:
        builder.abort()
        raise


def _read_file(path: str | os.PathLike[str], *, verify_checksum: bool) -> tuple[Path, dict[str, Any], NbnetFileInfo]:
    file_path = Path(path)
    file_bytes = file_path.stat().st_size
    with file_path.open("rb") as stream:
        raw_header = stream.read(_HEADER.size)
        if len(raw_header) != _HEADER.size:
            raise ValueError("truncated nbnet header")
        magic, major, minor, flags, record_size, metadata_size, connection_count, expected_digest = _HEADER.unpack(
            raw_header
        )
        if magic != _MAGIC:
            raise ValueError("invalid nbnet magic")
        if major != _VERSION_MAJOR:
            raise ValueError(f"unsupported nbnet major version: {major}")
        if record_size != _CONNECTION.size:
            raise ValueError(f"unsupported connection record size: {record_size}")
        if metadata_size > _MAX_METADATA_BYTES:
            raise ValueError(f"nbnet metadata exceeds {_MAX_METADATA_BYTES} bytes")
        expected_size = _HEADER.size + metadata_size + connection_count * record_size
        if file_bytes != expected_size:
            raise ValueError(f"nbnet size mismatch: expected {expected_size}, got {file_bytes}")
        metadata_bytes = stream.read(metadata_size)
        if len(metadata_bytes) != metadata_size:
            raise ValueError("truncated nbnet metadata")
        if verify_checksum:
            if not flags & _FLAG_SHA256:
                raise ValueError("nbnet file does not contain a supported checksum")
            digest = hashlib.sha256(metadata_bytes)
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            if digest.digest() != expected_digest:
                raise ValueError("nbnet checksum mismatch")
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("layers", []), list):
        raise ValueError("nbnet metadata must contain a layers list")
    try:
        neuron_count = sum(int(layer["count"]) for layer in metadata.get("layers", []))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid nbnet layer metadata") from exc
    info = NbnetFileInfo(
        path=file_path.resolve(),
        version=(major, minor),
        file_bytes=file_bytes,
        metadata_bytes=metadata_size,
        neuron_count=neuron_count,
        connection_count=connection_count,
        checksum=expected_digest.hex(),
    )
    return file_path, metadata, info


def inspect(path: str | os.PathLike[str], *, verify_checksum: bool = False) -> NbnetFileInfo:
    return _read_file(path, verify_checksum=verify_checksum)[2]


def validate(path: str | os.PathLike[str], *, verify_checksum: bool = True) -> NbnetValidationReport:
    try:
        info = inspect(path, verify_checksum=verify_checksum)
        return NbnetValidationReport(True, info, ())
    except (OSError, ValueError, TypeError, KeyError, UnicodeDecodeError, struct.error) as exc:
        return NbnetValidationReport(False, None, (str(exc),))


def is_valid(path: str | os.PathLike[str], *, verify_checksum: bool = True) -> bool:
    return validate(path, verify_checksum=verify_checksum).valid


__all__ = [
    "NbnetBuildResult",
    "NbnetDescriptionBuilder",
    "NbnetFileInfo",
    "NbnetLayerHandle",
    "NbnetValidationReport",
    "NbnetWriteOptions",
    "StreamingBuildOptions",
    "from_batches",
    "from_network",
    "inspect",
    "is_valid",
    "validate",
]


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m neuronbridge.nbnet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "validate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("path", type=Path)
        command_parser.add_argument("--no-checksum", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        result: Any = inspect(args.path, verify_checksum=not args.no_checksum)
    else:
        result = validate(args.path, verify_checksum=not args.no_checksum)
    print(json.dumps({field: getattr(result, field) for field in result.__dataclass_fields__}, default=str, indent=2))
    return 0 if getattr(result, "valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
