import neuronbridge as nb


def test_python_runs_native_lif_network():
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(
        nb.NeuronLayer(
            "TimeDrivenLIF_Exponential_double",
            1,
            output=True,
            parameters={
                "V_rest": nb.float32(-60.0),
                "V_reset": nb.float32(-60.0),
                "V_th": nb.float32(-50.0),
                "tau": nb.float32(20.0),
                "R": nb.float32(1.0),
                "t_ref": nb.int32(1),
                "gexc_tau": nb.float32(5.0),
                "ginh_tau": nb.float32(10.0),
                "Eexc": nb.float32(0.0),
                "Einh": nb.float32(-80.0),
            },
        )
    )
    network.connect(
        nb.Connection(
            source=0,
            target=1,
            weight=3.0,
            max_weight=3.0,
            delay=1,
        )
    )

    simulation = nb.Simulation(
        network,
        nb.SimulationConfig(steps=20, timestep=0.1),
    )
    simulation.init()
    assert simulation.initialized
    simulation.add_external_spikes([1], [0])
    simulation.run()
    assert isinstance(simulation.output_spikes(), list)
    assert simulation.get_connection_weight(0) == 3.0
