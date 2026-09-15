"""Reusable builders for EI benchmark migration examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import neuronbridge as nb
from _example_paths import data_dir


EXCITATORY_COUNT = 3200
INHIBITORY_COUNT = 800
NEURON_COUNT = EXCITATORY_COUNT + INHIBITORY_COUNT
CURRENT_NEURON_COUNT = 1
SIMULATION_STEPS = 2000
BASE_TIMESTEP_MS = 0.1
EXTERNAL_CURRENT = 12.0
EXC_WEIGHT = 0.3
INH_WEIGHT = 3.7
MAX_WEIGHT = 10.0
DELAY_STEPS = 1


@dataclass(slots=True)
class ConnectivityData:
    e2e_pre: list[int]
    e2e_post: list[int]
    e2i_pre: list[int]
    e2i_post: list[int]
    i2e_pre: list[int]
    i2e_post: list[int]
    i2i_pre: list[int]
    i2i_post: list[int]


def default_connectivity_dir() -> Path:
    return data_dir("ei_connectivity")


def read_ints(path: Path) -> list[int]:
    try:
        return [int(value) for value in path.read_text().split()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"connectivity file is missing: {path}") from exc


def load_connectivity(base_dir: Path | str | None = None) -> ConnectivityData:
    base = Path(base_dir) if base_dir is not None else default_connectivity_dir()
    data = ConnectivityData(
        e2e_pre=read_ints(base / "pre_ids_E2E.txt"),
        e2e_post=read_ints(base / "post_ids_E2E.txt"),
        e2i_pre=read_ints(base / "pre_ids_E2I.txt"),
        e2i_post=read_ints(base / "post_ids_E2I.txt"),
        i2e_pre=read_ints(base / "pre_ids_I2E.txt"),
        i2e_post=read_ints(base / "post_ids_I2E.txt"),
        i2i_pre=read_ints(base / "pre_ids_I2I.txt"),
        i2i_post=read_ints(base / "post_ids_I2I.txt"),
    )
    for label, pre, post in (
        ("E2E", data.e2e_pre, data.e2e_post),
        ("E2I", data.e2i_pre, data.e2i_post),
        ("I2E", data.i2e_pre, data.i2e_post),
        ("I2I", data.i2i_pre, data.i2i_post),
    ):
        if len(pre) != len(post):
            raise ValueError(f"{label} connectivity pre/post file sizes do not match")
    return data


def _lif_population(count: int, *, dense_name: str | None = None, output: bool = False) -> nb.NeuronLayer:
    return nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double",
        count,
        output=output,
        parameters={
            "V_rest": nb.float32(-60.0),
            "tau": nb.float32(20.0),
            "V_th": nb.float32(-50.0),
            "R": nb.float32(1.0),
            "V_reset": nb.float32(-60.0),
            "t_ref": nb.int32(50),
            "gexc_tau": nb.float32(5.0),
            "Eexc": nb.float32(0.0),
            "ginh_tau": nb.float32(10.0),
            "Einh": nb.float32(-80.0),
            "random_sigma": nb.float32_array4([0.0, 0.0, 0.0, 0.0]),
            **(
                {
                    "dense_subnetwork_name": dense_name,
                    "dense_steps_to_keep": nb.int32(8),
                    "dense_dt_ms": nb.float32(BASE_TIMESTEP_MS),
                }
                if dense_name is not None
                else {}
            ),
        },
    )


def _append_connection(
    network: nb.Network,
    pre_ids: list[int],
    post_ids: list[int],
    *,
    pre_offset: int,
    post_offset: int,
    synapse_type: int,
    weight: float,
) -> None:
    network.connect(
        nb.Connection(
            source=[pre + pre_offset for pre in pre_ids],
            target=[post + post_offset for post in post_ids],
            synapse_type=synapse_type,
            weight=weight,
            max_weight=MAX_WEIGHT,
            delay=DELAY_STEPS,
        )
    )


def _append_current_to_all(network: nb.Network) -> None:
    network.connect(
        nb.Connection(
            source=NEURON_COUNT,
            target=list(range(NEURON_COUNT)),
            synapse_type=3,
            weight=1.0,
            max_weight=MAX_WEIGHT,
            delay=DELAY_STEPS,
        )
    )


def build_dense_ei_program(connectivity: ConnectivityData) -> nb.Network:
    dense_name = "ei_program_dense_subnetwork"
    network = nb.Network()
    network.add_layer(_lif_population(EXCITATORY_COUNT, dense_name=dense_name, output=False))
    network.add_layer(_lif_population(INHIBITORY_COUNT, dense_name=dense_name, output=False))
    network.add_layer(nb.NeuronLayer.input_current(CURRENT_NEURON_COUNT))
    _append_connection(
        network,
        connectivity.e2e_pre,
        connectivity.e2e_post,
        pre_offset=0,
        post_offset=0,
        synapse_type=0,
        weight=EXC_WEIGHT,
    )
    _append_connection(
        network,
        connectivity.e2i_pre,
        connectivity.e2i_post,
        pre_offset=0,
        post_offset=EXCITATORY_COUNT,
        synapse_type=0,
        weight=EXC_WEIGHT,
    )
    _append_connection(
        network,
        connectivity.i2e_pre,
        connectivity.i2e_post,
        pre_offset=EXCITATORY_COUNT,
        post_offset=0,
        synapse_type=1,
        weight=INH_WEIGHT,
    )
    _append_connection(
        network,
        connectivity.i2i_pre,
        connectivity.i2i_post,
        pre_offset=EXCITATORY_COUNT,
        post_offset=EXCITATORY_COUNT,
        synapse_type=1,
        weight=INH_WEIGHT,
    )
    _append_current_to_all(network)
    return network


def build_main_ei_program(connectivity: ConnectivityData) -> nb.Network:
    network = nb.Network()
    network.add_layer(_lif_population(EXCITATORY_COUNT, output=True))
    network.add_layer(_lif_population(INHIBITORY_COUNT, output=True))
    network.add_layer(nb.NeuronLayer.input_current(CURRENT_NEURON_COUNT))
    _append_connection(
        network,
        connectivity.e2e_pre,
        connectivity.e2e_post,
        pre_offset=0,
        post_offset=0,
        synapse_type=0,
        weight=EXC_WEIGHT,
    )
    _append_connection(
        network,
        connectivity.e2i_pre,
        connectivity.e2i_post,
        pre_offset=0,
        post_offset=EXCITATORY_COUNT,
        synapse_type=0,
        weight=EXC_WEIGHT,
    )
    _append_connection(
        network,
        connectivity.i2e_pre,
        connectivity.i2e_post,
        pre_offset=EXCITATORY_COUNT,
        post_offset=0,
        synapse_type=1,
        weight=INH_WEIGHT,
    )
    _append_connection(
        network,
        connectivity.i2i_pre,
        connectivity.i2i_post,
        pre_offset=EXCITATORY_COUNT,
        post_offset=EXCITATORY_COUNT,
        synapse_type=1,
        weight=INH_WEIGHT,
    )
    _append_current_to_all(network)
    return network
