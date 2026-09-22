"""Dependency-free smoke test for an installed wheel's .nbnet path."""

from __future__ import annotations

from pathlib import Path
import tempfile

import neuronbridge as nb


def main() -> None:
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
    network.connect(nb.Connection(source=0, target=1, weight=3.0, max_weight=3.0, delay=1))
    config = nb.SimulationConfig(steps=20, timestep=0.1)

    with tempfile.TemporaryDirectory(prefix="neuronbridge-nbnet-") as directory:
        path = Path(directory) / "smoke.nbnet"
        result = nb.nbnet.from_network(network, path)
        assert result.connection_count == 1
        assert nb.nbnet.validate(path).valid
        fast = nb.Simulation(network, config).init()
        streaming = nb.Simulation(path, config).init()
        fast.add_external_spikes([1], [0]).run()
        streaming.add_external_spikes([1], [0]).run()
        assert fast.output_spikes() == streaming.output_spikes()
        assert abs(fast.get_connection_weight(0) - streaming.get_connection_weight(0)) <= 1e-7
        assert streaming.build_stats["mode"] == "streaming_file_cpp"
        assert streaming.build_stats["runtime_build_path"] == "streaming_direct"

    module_path = Path(nb.__file__).resolve()
    print(f"installed wheel nbnet smoke PASS: {module_path}")


if __name__ == "__main__":
    main()
