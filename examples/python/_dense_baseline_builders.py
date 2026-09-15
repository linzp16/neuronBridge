"""Reusable network builders for migrated dense baseline examples."""

from __future__ import annotations

import neuronbridge as nb


def all_to_all(source_begin, source_count, target_begin, target_count, *, weight, delay=1, synapse_type=0):
    count = source_count * target_count
    return nb.Connection(
        source=[source_begin + source for source in range(source_count) for _ in range(target_count)],
        target=[target_begin + target for _ in range(source_count) for target in range(target_count)],
        synapse_type=[int(synapse_type)] * count,
        weight=[float(weight)] * count,
        max_weight=[10.0] * count,
        delay=[int(delay)] * count,
    )


def lif_double_layer(
    count: int,
    *,
    update_timestep: int = 1,
    monitored: bool = False,
    output: bool = False,
    dense_name: str | None = None,
    random_sigma: list[float] | None = None,
    **parameters,
) -> nb.NeuronLayer:
    params = dict(parameters)
    if dense_name is not None:
        params["dense_subnetwork_name"] = dense_name
    if random_sigma is not None:
        params["random_sigma"] = nb.float32_array4(random_sigma)
    return nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double",
        count,
        update_timestep=update_timestep,
        monitored=monitored,
        output=output,
        parameters=params,
    )


def lif_decay_layer(
    count: int,
    *,
    monitored: bool = False,
    output: bool = False,
    random_sigma: list[float] | None = None,
    **parameters,
) -> nb.NeuronLayer:
    params = dict(parameters)
    if random_sigma is not None:
        params["random_sigma"] = nb.float32_array3(random_sigma)
    return nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_Decay",
        count,
        monitored=monitored,
        output=output,
        parameters=params,
    )


def build_dense_mixed_network(*, connect_relay_to_dense: bool = True) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(
        lif_double_layer(
            2,
            monitored=True,
            random_sigma=[1.0, 0.0, 0.0, 0.0],
            t_ref=nb.int32(2),
        )
    )
    network.add_layer(
        lif_double_layer(
            4,
            update_timestep=1,
            monitored=True,
            output=False,
            dense_name="mixed_dense_subnetwork",
            random_sigma=[2.0, 0.0, 0.0, 0.0],
            t_ref=nb.int32(2),
        )
    )
    network.add_layer(
        lif_double_layer(
            3,
            update_timestep=2,
            monitored=True,
            output=False,
            dense_name="mixed_dense_subnetwork",
            random_sigma=[3.0, 0.0, 0.0, 0.0],
            t_ref=nb.int32(2),
        )
    )
    network.add_layer(
        lif_decay_layer(
            2,
            monitored=True,
            output=True,
            random_sigma=[0.0, 0.0, 0.0],
            t_ref=nb.int32(2),
        )
    )
    network.connect(all_to_all(0, 1, 1, 2, weight=1.0, delay=1))
    if connect_relay_to_dense:
        network.connect(all_to_all(1, 2, 3, 4, weight=1.0, delay=1))
    network.connect(all_to_all(3, 4, 7, 3, weight=1.0, delay=2))
    network.connect(all_to_all(7, 3, 10, 2, weight=0.3, delay=3))
    return network


def build_dense_mixed_current_network() -> nb.Network:
    dense_name = "mixed_current_dense_demo"
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.input_current(1))
    for count, sigma in ((4, [1.0, 0.0, 0.0, 0.0]), (3, [2.0, 0.0, 0.0, 0.0])):
        network.add_layer(
            lif_double_layer(
                count,
                dense_name=dense_name,
                random_sigma=sigma,
                tau_m_ms=nb.float32(20.0),
                tau_exc_ms=nb.float32(5.0),
                tau_inh_ms=nb.float32(10.0),
                v_rest=nb.float32(-65.0),
                v_reset=nb.float32(-65.0),
                v_threshold=nb.float32(-55.0),
            )
        )
    network.add_layer(lif_decay_layer(2, output=True, random_sigma=[0.0, 0.0, 0.0]))
    network.connect(all_to_all(0, 1, 2, 4, weight=6.0, delay=1))
    network.connect(all_to_all(1, 1, 2, 4, weight=1.0, delay=1, synapse_type=3))
    network.connect(all_to_all(2, 4, 6, 3, weight=5.0, delay=1))
    network.connect(all_to_all(6, 3, 9, 2, weight=4.0, delay=1))
    return network


def build_dense_run_no_debug_network() -> nb.Network:
    dense_name = "dense_run_no_debug_demo"
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    for count, sigma in ((4, [1.0, 0.0, 0.0, 0.0]), (3, [2.0, 0.0, 0.0, 0.0])):
        network.add_layer(
            lif_double_layer(
                count,
                dense_name=dense_name,
                random_sigma=sigma,
                tau=nb.float32(20.0),
                gexc_tau=nb.float32(5.0),
                ginh_tau=nb.float32(10.0),
                V_rest=nb.float32(-65.0),
                V_reset=nb.float32(-65.0),
                V_th=nb.float32(-55.0),
                Eexc=nb.float32(0.0),
                Einh=nb.float32(-80.0),
                t_ref=nb.int32(2),
            )
        )
    network.add_layer(lif_decay_layer(2, output=True, random_sigma=[0.0, 0.0, 0.0]))
    network.connect(all_to_all(0, 1, 1, 4, weight=2.0, delay=1))
    network.connect(all_to_all(1, 4, 5, 3, weight=1.0, delay=1))
    network.connect(all_to_all(5, 3, 8, 2, weight=1.0, delay=1))
    return network
