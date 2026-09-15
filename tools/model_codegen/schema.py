"""Shared, model-agnostic schema objects for maintained code generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


class SchemaError(ValueError):
    """A model declaration is malformed independently of any backend."""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    api_name: str
    default: float | int
    storage: str = "float32"
    aliases: tuple[str, ...] = ()
    exclusive_min: float | None = None
    inclusive_min: float | None = None
    inclusive_max: float | None = None


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaError(f"{label} must be an identifier")
    return value


def compile_parameters(raw: object) -> tuple[ParameterSpec, ...]:
    if not isinstance(raw, dict):
        raise SchemaError("parameters must be an object")
    result: list[ParameterSpec] = []
    api_names: set[str] = set()
    for name, declaration in raw.items():
        require_identifier(name, "parameter name")
        if not isinstance(declaration, dict):
            raise SchemaError(f"parameter {name!r} must be an object")
        unknown = set(declaration) - {
            "api_name", "aliases", "default", "storage", "constraints"
        }
        if unknown:
            raise SchemaError(f"parameter {name!r} has unsupported fields {sorted(unknown)!r}")
        api_name = declaration.get("api_name")
        if not isinstance(api_name, str) or not api_name:
            raise SchemaError(f"parameter {name!r} requires api_name")
        if api_name in api_names:
            raise SchemaError(f"duplicate parameter api_name {api_name!r}")
        api_names.add(api_name)
        aliases = declaration.get("aliases", [])
        if (
            not isinstance(aliases, list)
            or any(not isinstance(alias, str) or not alias for alias in aliases)
            or len(set(aliases)) != len(aliases)
            or api_name in aliases
        ):
            raise SchemaError(f"parameter {name!r} aliases must be unique strings")
        for alias in aliases:
            if alias in api_names:
                raise SchemaError(f"duplicate parameter API alias {alias!r}")
            api_names.add(alias)
        default = declaration.get("default")
        if type(default) not in (int, float) or not math.isfinite(default):
            raise SchemaError(f"parameter {name!r} requires a finite numeric default")
        storage = declaration.get("storage", "float32")
        if storage not in {"float32", "int32"}:
            raise SchemaError(f"parameter {name!r} has unsupported storage {storage!r}")
        if storage == "int32" and (not isinstance(default, int) or isinstance(default, bool)):
            raise SchemaError(f"parameter {name!r} int32 default must be an integer")
        default = int(default) if storage == "int32" else float(default)
        if storage == "float32" and abs(default) > 3.402823466e38:
            raise SchemaError(f"parameter {name!r} default is outside float32 range")
        constraints = declaration.get("constraints", {})
        if not isinstance(constraints, dict):
            raise SchemaError(f"parameter {name!r} constraints must be an object")
        if set(constraints) - {"exclusive_min", "inclusive_min", "inclusive_max"}:
            raise SchemaError(f"parameter {name!r} has unsupported constraints")
        values: dict[str, float | None] = {}
        for key in ("exclusive_min", "inclusive_min", "inclusive_max"):
            value = constraints.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise SchemaError(f"parameter {name!r} constraint {key!r} must be finite")
            values[key] = None if value is None else float(value)
        parameter = ParameterSpec(name, api_name, default, storage, tuple(aliases), **values)
        validate_parameter_default(parameter)
        result.append(parameter)
    return tuple(result)


def validate_parameter_default(parameter: ParameterSpec) -> None:
    value = parameter.default
    if parameter.exclusive_min is not None and value <= parameter.exclusive_min:
        raise SchemaError(f"parameter {parameter.name!r} must exceed its exclusive minimum")
    if parameter.inclusive_min is not None and value < parameter.inclusive_min:
        raise SchemaError(f"parameter {parameter.name!r} is below its inclusive minimum")
    if parameter.inclusive_max is not None and value > parameter.inclusive_max:
        raise SchemaError(f"parameter {parameter.name!r} exceeds its inclusive maximum")


def cpp_constraint(parameter: ParameterSpec, member: str) -> str:
    clauses = [f"!std::isfinite({member})"]
    if parameter.exclusive_min is not None:
        clauses.append(f"!({member} > {parameter.exclusive_min!r}f)")
    if parameter.inclusive_min is not None:
        clauses.append(f"{member} < {parameter.inclusive_min!r}f")
    if parameter.inclusive_max is not None:
        clauses.append(f"{member} > {parameter.inclusive_max!r}f")
    return " || ".join(clauses)
