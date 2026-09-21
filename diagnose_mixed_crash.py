import sys
import neuronbridge as nb

variant = sys.argv[1]
n = nb.Network()
n.add_layer(nb.NeuronLayer.input_spike(2))
if variant in {"current", "currentrun", "full", "fullrun", "fullspike", "fullcurrent", "fullcurrent_norules"}:
    n.add_layer(nb.NeuronLayer.input_current(2))
n.add_layer(nb.NeuronLayer.lif_decay(2, monitored=True))
if variant in {"izh", "full", "fullrun", "fullspike", "fullcurrent", "fullcurrent_norules"}:
    n.add_layer(nb.NeuronLayer("TimeDrivenIzhikevic_Exponential_Decay", 2, monitored=True))
if variant in {"poisson", "full", "fullrun", "fullspike", "fullcurrent", "fullcurrent_norules"}:
    n.add_layer(nb.NeuronLayer.poisson_rate(2, rate_bias_hz=20.0, rate_gain_hz_per_current=2.0))
if variant in {"trigger", "full", "fullrun", "fullspike", "fullcurrent", "fullcurrent_norules"}:
    n.add_layer(nb.NeuronLayer("TriggerRelayNeuronModel", 2, output=True))

if variant == "base":
    n.connect(nb.Connection(0, [2, 3], weight=0.5, delay=1))
elif variant in {"current", "currentrun"}:
    n.connect(nb.Connection(0, [4, 5], weight=0.5, delay=1))
elif variant == "izh":
    n.add_layer(nb.NeuronLayer("TimeDrivenIzhikevic_Exponential_Decay", 2))
    n.connect(nb.Connection(0, [2, 3], weight=0.5, delay=1))
    n.connect(nb.Connection(2, [4, 5], weight=0.4, delay=1))
elif variant == "poisson":
    n.add_layer(nb.NeuronLayer.poisson_rate(2, rate_bias_hz=20.0, rate_gain_hz_per_current=2.0))
    n.connect(nb.Connection(0, [2, 3], weight=0.5, delay=1))
    n.connect(nb.Connection(2, [4, 5], weight=0.2, delay=1))
elif variant == "trigger":
    n.add_layer(nb.NeuronLayer("TriggerRelayNeuronModel", 2, output=True))
    n.connect(nb.Connection(0, [2, 3], weight=0.5, delay=1))
    n.connect(nb.Connection(2, [4, 5], weight=1.0, delay=1))
else:
    n.connect(nb.Connection([0, 1], [4, 5], weight=[0.5, 0.7], delay=[1, 2]))
    n.connect(nb.Connection([2, 3], [6, 7], weight=0.4, delay=1))
    n.connect(nb.Connection([4, 5], [8, 9], weight=0.2, delay=1))
    n.connect(nb.Connection([6, 7], [10, 11], weight=1.0, delay=1))
    if variant != "fullcurrent_norules":
        n.add_learning_rule(nb.LearningRule("CustomPairStdpV1"))
        n.add_learning_rule(nb.LearningRule("CustomRStdpV1"))

print("defined", variant, flush=True)
queues = 2 if variant in {"full", "fullrun"} else 1
sim = nb.Simulation(n, nb.SimulationConfig(steps=10, timestep=1.0, queues=queues)).init()
print("initialized", variant, flush=True)
if variant in {"currentrun", "fullrun", "fullspike", "fullcurrent", "fullcurrent_norules"}:
    sim.add_external_spikes([0, 1, 5, 10], [0, 1, 0, 1])
    if variant in {"currentrun", "fullrun", "fullcurrent", "fullcurrent_norules"}:
        sim.add_external_currents([0, 2, 6], [2, 3, 2], [5.0, 8.0, 5.0])
    sim.run()
    print("ran", variant, flush=True)
