"""Lazy readers for DebugMonitor output directories."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence


class DebugMonitorResult:
    """Lazy access to CSV/JSON files emitted by the C++ DebugMonitor."""

    KNOWN_TABLES = {
        "neuron_state": "neuron_state.csv",
        "pending_channels": "pending_channels.csv",
        "spikes": "spikes.csv",
        "weights": "weights.csv",
        "outer_dynamic_state": "outer_dynamic_state.csv",
        "inputconv_outputs": "inputconv_outputs.csv",
        "inputconv_inputs": "inputconv_inputs.csv",
        "inputconv_state": "inputconv_state.csv",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    def meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def table_path(self, name: str) -> Path:
        filename = self.KNOWN_TABLES.get(name, name)
        path = self.path / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def iter_rows(self, name: str, *, limit: int | None = None) -> Iterator[dict[str, str]]:
        """Iterate rows without loading large monitor files into memory."""
        count = 0
        with self.table_path(name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
                count += 1
                if limit is not None and count >= limit:
                    return

    def to_pandas(self, name: str, **kwargs):
        """Load one table with pandas when the caller explicitly asks for it."""
        import pandas as pd

        return pd.read_csv(self.table_path(name), **kwargs)

    def file_sizes(self) -> dict[str, int]:
        sizes = {}
        for table, filename in self.KNOWN_TABLES.items():
            path = self.path / filename
            if path.exists():
                sizes[table] = path.stat().st_size
        return sizes

    def plot_spike_raster(self, *, ax=None, limit: int | None = None, **scatter_kwargs):
        """Draw a spike raster from `spikes.csv`."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        times = []
        neurons = []
        for row in self.iter_rows("spikes", limit=limit):
            times.append(int(row.get("time_step", 0)))
            neurons.append(int(row.get("global_neuron_id", row.get("local_neuron_id", -1))))
        options = {"s": 4, "linewidths": 0}
        options.update(scatter_kwargs)
        ax.scatter(times, neurons, **options)
        ax.set_xlabel("time step")
        ax.set_ylabel("neuron id")
        return ax

    def plot_state_trace(self, neuron_id: int, field_name: str = "voltage", *, ax=None, limit: int | None = None):
        """Draw one neuron-state field over time from `neuron_state.csv`."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        times = []
        values = []
        for row in self.iter_rows("neuron_state", limit=limit):
            global_id = int(row.get("global_neuron_id", -1))
            if global_id != neuron_id or row.get("field_name") != field_name:
                continue
            times.append(int(row.get("time_step", 0)))
            values.append(float(row.get("value", 0.0)))
        ax.plot(times, values)
        ax.set_xlabel("time step")
        ax.set_ylabel(field_name)
        return ax

    def plot_weight_trace(
        self,
        component: str | tuple[str, int] | None,
        synapse_ids: int | Iterable[int],
        *,
        ax=None,
        limit: int | None = None,
        label_template: str = "{component}:{synapse_id}",
        **plot_kwargs,
    ):
        """Draw selected synaptic weights over monitor time steps.

        ``component`` is a component name or a ``(component_kind, component_index)``
        pair. Requiring it avoids collisions between main-network and dense-runtime
        synapse identifiers.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(constrained_layout=True)
        selected_ids = self._normalize_indices(synapse_ids, name="synapse_ids")
        series = {synapse_id: ([], []) for synapse_id in selected_ids}
        for row in self.iter_rows("weights", limit=limit):
            if not self._matches_component(row, component):
                continue
            synapse_id = int(row["synapse_id"])
            if synapse_id not in series:
                continue
            times, values = series[synapse_id]
            times.append(int(row["time_step"]))
            values.append(float(row["value"]))
        missing = [str(synapse_id) for synapse_id, (times, _) in series.items() if not times]
        if missing:
            raise ValueError(f"no weight samples found for synapse_ids: {', '.join(missing)}")
        for synapse_id, (times, values) in series.items():
            ax.plot(
                times,
                values,
                label=label_template.format(component=self._component_label(component), synapse_id=synapse_id),
                **plot_kwargs,
            )
        ax.set_xlabel("time step")
        ax.set_ylabel("weight")
        ax.legend()
        return ax

    def plot_weight_summary(
        self,
        component: str | tuple[str, int] | None,
        *,
        synapse_ids: int | Iterable[int] | None = None,
        statistics: Sequence[str] = ("mean", "std"),
        band: bool = True,
        ax=None,
        limit: int | None = None,
        **plot_kwargs,
    ):
        """Plot streaming per-time-step weight statistics for one component."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(constrained_layout=True)
        allowed = {"mean", "std", "min", "max"}
        selected_stats = tuple(statistics)
        invalid = set(selected_stats) - allowed
        if not selected_stats or invalid:
            raise ValueError(f"statistics must be a non-empty subset of {sorted(allowed)}")
        selected_ids = None if synapse_ids is None else set(self._normalize_indices(synapse_ids, name="synapse_ids"))
        aggregates: dict[int, list[float]] = {}
        for row in self.iter_rows("weights", limit=limit):
            if not self._matches_component(row, component):
                continue
            if selected_ids is not None and int(row["synapse_id"]) not in selected_ids:
                continue
            time_step = int(row["time_step"])
            value = float(row["value"])
            aggregate = aggregates.setdefault(time_step, [0.0, 0.0, 0.0, value, value])
            aggregate[0] += 1.0
            aggregate[1] += value
            aggregate[2] += value * value
            aggregate[3] = min(aggregate[3], value)
            aggregate[4] = max(aggregate[4], value)
        if not aggregates:
            raise ValueError("no weight samples found for the requested component and synapse selection")
        times = sorted(aggregates)
        means = [aggregates[time_step][1] / aggregates[time_step][0] for time_step in times]
        stds = [
            math.sqrt(max(0.0, aggregates[time_step][2] / aggregates[time_step][0] - mean * mean))
            for time_step, mean in zip(times, means)
        ]
        minima = [aggregates[time_step][3] for time_step in times]
        maxima = [aggregates[time_step][4] for time_step in times]
        values_by_stat = {"mean": means, "std": stds, "min": minima, "max": maxima}
        for stat in selected_stats:
            if stat == "std" and band and "mean" in selected_stats:
                continue
            ax.plot(times, values_by_stat[stat], label=stat, **plot_kwargs)
        if band and "mean" in selected_stats and "std" in selected_stats:
            lower = [mean - std for mean, std in zip(means, stds)]
            upper = [mean + std for mean, std in zip(means, stds)]
            ax.fill_between(times, lower, upper, alpha=0.2, label="mean +/- std")
        ax.set_xlabel("time step")
        ax.set_ylabel("weight")
        ax.legend()
        return ax

    def plot_weight_distribution(
        self,
        component: str | tuple[str, int] | None,
        *,
        time_step: int | None = None,
        synapse_ids: int | Iterable[int] | None = None,
        bins: int = 80,
        ax=None,
        limit: int | None = None,
        **hist_kwargs,
    ):
        """Plot a weight histogram at one monitor time step, defaulting to the last."""
        if bins <= 0:
            raise ValueError("bins must be positive")
        selected_ids = None if synapse_ids is None else set(self._normalize_indices(synapse_ids, name="synapse_ids"))
        rows = []
        latest_time_step = None
        for row in self.iter_rows("weights", limit=limit):
            if not self._matches_component(row, component):
                continue
            if selected_ids is not None and int(row["synapse_id"]) not in selected_ids:
                continue
            rows.append(row)
            latest_time_step = max(latest_time_step, int(row["time_step"])) if latest_time_step is not None else int(row["time_step"])
        selected_time_step = latest_time_step if time_step is None else int(time_step)
        values = [float(row["value"]) for row in rows if int(row["time_step"]) == selected_time_step]
        if not values:
            raise ValueError(f"no weight samples found at time_step={selected_time_step}")
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(constrained_layout=True)
        options = {"edgecolor": "white", "alpha": 0.85}
        options.update(hist_kwargs)
        ax.hist(values, bins=bins, **options)
        ax.set_xlabel("weight")
        ax.set_ylabel("synapse count")
        ax.set_title(f"weight distribution at time step {selected_time_step}")
        return ax

    def plot_outer_dynamic_trace(
        self,
        fields: Sequence[str] = ("q", "q_des"),
        *,
        outer_dynamic: str | int | None = None,
        indices: int | Iterable[int] = (0, 1),
        ax=None,
        layout: str = "overlay",
        limit: int | None = None,
        **plot_kwargs,
    ):
        """Plot monitored OuterDynamic state fields for the current single-instance schema."""
        selected_fields = tuple(fields)
        if not selected_fields:
            raise ValueError("fields must not be empty")
        selected_indices = self._normalize_indices(indices, name="indices")
        if layout not in {"overlay", "stacked"}:
            raise ValueError("layout must be 'overlay' or 'stacked'")
        if ax is not None and layout != "overlay":
            raise ValueError("an explicit ax is only supported with layout='overlay'")
        series = self._outer_dynamic_series(selected_fields, selected_indices, limit, outer_dynamic)
        if ax is not None:
            axes = [ax]
        else:
            import matplotlib.pyplot as plt

            count = 1 if layout == "overlay" else len(selected_fields)
            _, axes_grid = plt.subplots(count, 1, sharex=True, squeeze=False, constrained_layout=True)
            axes = [axes_grid[index][0] for index in range(count)]
        for field_position, field_name in enumerate(selected_fields):
            target_ax = axes[0] if layout == "overlay" else axes[field_position]
            for index in selected_indices:
                times, values = series[(field_name, index)]
                target_ax.plot(times, values, label=f"{field_name}[{index}]", **plot_kwargs)
            target_ax.set_ylabel(field_name if layout == "stacked" else "value")
            target_ax.legend()
        axes[-1].set_xlabel("time step")
        return axes[0] if ax is not None or layout == "overlay" else axes

    def plot_outer_dynamic_tracking_error(
        self,
        field: str = "q",
        *,
        outer_dynamic: str | int | None = None,
        desired_field: str | None = None,
        indices: int | Iterable[int] = (0, 1),
        ax=None,
        limit: int | None = None,
        **plot_kwargs,
    ):
        """Plot actual-minus-desired OuterDynamic state errors."""
        desired_field = desired_field or f"{field}_des"
        selected_indices = self._normalize_indices(indices, name="indices")
        series = self._outer_dynamic_series((field, desired_field), selected_indices, limit, outer_dynamic)
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(constrained_layout=True)
        for index in selected_indices:
            actual = dict(zip(*series[(field, index)]))
            desired = dict(zip(*series[(desired_field, index)]))
            times = sorted(set(actual) & set(desired))
            if not times:
                raise ValueError(f"no aligned samples for {field}[{index}] and {desired_field}[{index}]")
            ax.plot(times, [actual[time_step] - desired[time_step] for time_step in times], label=f"{field}[{index}] error", **plot_kwargs)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("time step")
        ax.set_ylabel(f"{field} error")
        ax.legend()
        return ax

    def plot_outer_dynamic_phase_plane(
        self,
        joint_index: int = 0,
        *,
        outer_dynamic: str | int | None = None,
        position_field: str = "q",
        velocity_field: str = "qv",
        ax=None,
        limit: int | None = None,
        **plot_kwargs,
    ):
        """Plot position against velocity for one OuterDynamic joint."""
        series = self._outer_dynamic_series(
            (position_field, velocity_field),
            [int(joint_index)],
            limit,
            outer_dynamic,
        )
        position = dict(zip(*series[(position_field, int(joint_index))]))
        velocity = dict(zip(*series[(velocity_field, int(joint_index))]))
        times = sorted(set(position) & set(velocity))
        if not times:
            raise ValueError(f"no aligned samples for {position_field} and {velocity_field}")
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(constrained_layout=True)
        ax.plot([position[time_step] for time_step in times], [velocity[time_step] for time_step in times], **plot_kwargs)
        ax.set_xlabel(f"{position_field}[{joint_index}]")
        ax.set_ylabel(f"{velocity_field}[{joint_index}]")
        return ax

    def plot_outer_dynamic_torque(
        self,
        *,
        outer_dynamic: str | int | None = None,
        indices: int | Iterable[int] = (0, 1),
        ax=None,
        limit: int | None = None,
        **plot_kwargs,
    ):
        """Plot total torque for selected OuterDynamic joints."""
        result = self.plot_outer_dynamic_trace(
            ("tau_total",),
            outer_dynamic=outer_dynamic,
            indices=indices,
            ax=ax,
            limit=limit,
            **plot_kwargs,
        )
        result.set_ylabel("total torque")
        return result

    @staticmethod
    def _normalize_indices(values: int | Iterable[int], *, name: str) -> tuple[int, ...]:
        if isinstance(values, int):
            normalized = (int(values),)
        else:
            normalized = tuple(int(value) for value in values)
        if not normalized:
            raise ValueError(f"{name} must not be empty")
        return normalized

    @staticmethod
    def _matches_component(row: dict[str, str], component: str | tuple[str, int] | None) -> bool:
        if component is None:
            return True
        if isinstance(component, str):
            return row.get("component_name") == component
        kind, index = component
        return row.get("component_kind") == kind and int(row.get("component_index", -1)) == int(index)

    @staticmethod
    def _component_label(component: str | tuple[str, int] | None) -> str:
        if component is None:
            return "all components"
        if isinstance(component, str):
            return component
        return f"{component[0]}[{component[1]}]"

    def _outer_dynamic_series(
        self,
        fields: Sequence[str],
        indices: Sequence[int],
        limit: int | None,
        outer_dynamic: str | int | None,
    ) -> dict[tuple[str, int], tuple[list[int], list[float]]]:
        series = {(field, index): ([], []) for field in fields for index in indices}
        for row in self.iter_rows("outer_dynamic_state", limit=limit):
            if not self._matches_outer_dynamic(row, outer_dynamic):
                continue
            key = (row["field_name"], int(row["index"]))
            if key not in series:
                continue
            times, values = series[key]
            times.append(int(row["time_step"]))
            values.append(float(row["value"]))
        missing = [f"{field}[{index}]" for (field, index), (times, _) in series.items() if not times]
        if missing:
            raise ValueError(f"no OuterDynamic samples found for: {', '.join(missing)}")
        return series

    @staticmethod
    def _matches_outer_dynamic(row: dict[str, str], outer_dynamic: str | int | None) -> bool:
        if outer_dynamic is None:
            return True
        if "component_name" not in row:
            return outer_dynamic == "outer_dynamic" or outer_dynamic == 0
        if isinstance(outer_dynamic, str):
            return row.get("component_name") == outer_dynamic
        return int(row.get("component_index", -1)) == int(outer_dynamic)
