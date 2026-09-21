import sys
import neuronbridge as nb

variant = sys.argv[1]
n = nb.Network()
n.add_layer(nb.NeuronLayer.input_spike(2))
n.add_layer(nb.NeuronLayer.input_current(2))
n.add_layer(nb.NeuronLayer.lif_decay(2))
next_id = 6
if "izh" in variant:
    n.add_layer(nb.NeuronLayer("TimeDrivenIzhikevic_Exponential_Decay", 2))
    n.connect(nb.Connection(2, 6, weight=0.4, delay=1))
    next_id += 2
if "poisson" in variant:
    n.add_layer(nb.NeuronLayer.poisson_rate(2, rate_bias_hz=20.0, rate_gain_hz_per_current=2.0))
    n.connect(nb.Connection(2, next_id, weight=0.2, delay=1))
    next_id += 2
if "trigger" in variant:
    n.add_layer(nb.NeuronLayer("TriggerRelayNeuronModel", 2, output=True))
    n.connect(nb.Connection(2, next_id, weight=1.0, delay=1))
n.connect(nb.Connection(0, 4, weight=0.5, delay=1))
n.connect(nb.Connection(2, 4, weight=0.5, delay=1))
print("defined", variant, flush=True)
sim = nb.Simulation(n, nb.SimulationConfig(steps=10, timestep=1.0, queues=1)).init()
print("initialized", variant, flush=True)
sim.add_external_currents([0, 2, 6], [2, 3, 2], [5.0, 8.0, 5.0]).run()
print("ran", variant, flush=True)
