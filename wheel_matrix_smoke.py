from pathlib import Path
import json
import neuronbridge as nb

root = Path(r"D:\testfile\05_simulation_modes")
rows = []

def run_case(name, config):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(1, output=True, monitored=True))
    network.connect(nb.Connection(source=0, target=1, weight=1.0, delay=1))
    sim = nb.Simulation(network, config).init()
    sim.add_external_spikes([0, 1, 2], [0, 0, 0]).run()
    rows.append({"case": name, "ok": True, "spikes": sim.output_spikes(), "states": sim.neuron_states([1])})

for name, cfg in [
    ("heap_single_queue", nb.SimulationConfig(steps=5, timestep=1.0, queues=1, event_queue="heap")),
    ("timing_wheel_single_queue", nb.SimulationConfig(steps=5, timestep=1.0, queues=1, event_queue="timing_wheel", timing_wheel_size=64)),
    ("heap_multi_queue", nb.SimulationConfig(steps=5, timestep=1.0, queues=2, event_queue="heap")),
]:
    run_case(name, cfg)

definitions = {
    "bar": nb.InputConv.v1_bar(width=2, height=2, channels=1, output_source_indices=[0], output_target_neuron_ids=[0]),
    "grating": nb.InputConv.v1_grating(width=2, height=2, channels=1, output_source_indices=[0], output_target_neuron_ids=[0]),
    "plaid": nb.InputConv.v1_plaid(width=2, height=2, channels=1, output_source_indices=[0], output_target_neuron_ids=[0]),
    "file": nb.InputConv.v1_file(stimulus_file_path="stimulus.bin", width=2, height=2, channels=1, output_source_indices=[0], output_target_neuron_ids=[0]),
}
for name, value in definitions.items():
    assert value.to_dict()["parameters"]["stimulus_mode"] == name

outer = nb.OuterDynamic.spike_counter(slot_count=1)
planar = nb.OuterDynamic.planar_arm_2dof()
strict = nb.OuterDynamic.strict_matlab_planar_arm_2dof()
result = {"simulation_cases": rows, "inputconv_modes": list(definitions), "outer_dynamic_models": [outer.model, planar.model, strict.model], "ok": True}
(root / "wheel_matrix_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
