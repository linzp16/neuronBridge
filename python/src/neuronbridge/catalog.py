"""Runtime model discovery backed by the native C++ catalogs and factories."""

from __future__ import annotations


def _native():
    from . import _core

    if _core is None:
        raise RuntimeError("neuronbridge native extension is not built")
    return _core


def list_models(*, backend: str | None = None, public_only: bool = True) -> list[str]:
    """Return canonical neuron model names compiled into this build."""
    return list(_native().list_neuron_models(backend or "", bool(public_only)))


def describe_model(
    name: str,
    *,
    backend: str | None = None,
    parameters: dict | None = None,
    timestep_size: int = 1,
    base_timestep: float = 1.0,
) -> dict:
    """Return backend support and parameters reported by the native model."""
    if timestep_size <= 0:
        raise ValueError("timestep_size must be positive")
    if base_timestep <= 0:
        raise ValueError("base_timestep must be positive")
    return dict(
        _native().describe_neuron_model(
            str(name),
            backend or "",
            {} if parameters is None else dict(parameters),
            int(timestep_size),
            float(base_timestep),
        )
    )
