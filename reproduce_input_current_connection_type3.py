import neuronbridge as nb

network = nb.Network()
network.add_layer(nb.NeuronLayer.input_current(2))
network.add_layer(nb.NeuronLayer.lif_decay(2, output=True))
network.connect(nb.Connection(0, [2, 3], synapse_type=3, weight=0.5, delay=1))
print("defined", flush=True)
simulation = nb.Simulation(network, nb.SimulationConfig(steps=10, timestep=1.0)).init()
print("initialized", flush=True)
simulation.add_external_currents([0, 2, 6], [0, 1, 0], [5.0, 8.0, 5.0]).run()
print("completed", flush=True)
