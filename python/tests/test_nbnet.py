from __future__ import annotations

from pathlib import Path

import pytest

import neuronbridge as nb


def _lif_layer() -> nb.NeuronLayer:
    return nb.NeuronLayer(
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


def _network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(_lif_layer())
    network.connect(nb.Connection(source=0, target=1, weight=3.0, max_weight=3.0, delay=1))
    return network


def _additive_kernel_network(*, dense: bool = False) -> tuple[nb.Network, int]:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2 if not dense else 1))
    network.add_learning_rule(
        nb.LearningRule(
            "AdditiveKernalChange",
            parameters={
                "a1pre": nb.float32(0.2),
                "a2prepre": nb.float32(-0.05),
                "LTP_tau": nb.float32(16.8),
            },
        )
    )
    if not dense:
        network.add_layer(_lif_layer())
        network.connect(
            nb.Connection(source=0, target=2, weight=8.0, max_weight=20.0, delay=1, synapse_rule=0)
        )
        network.connect(
            nb.Connection(source=1, target=2, weight=0.0, max_weight=0.0, delay=1, trigger_rule=0)
        )
        return network, 0

    network.add_layer(
        nb.NeuronLayer.lif_double(
            2,
            dense_name="additive_dense",
            monitored=True,
            V_th=nb.float32(-58.0),
        )
    )
    network.add_layer(
        nb.NeuronLayer(
            "TriggerRelayNeuronModel",
            1,
            parameters={"dense_subnetwork_name": "additive_dense"},
        )
    )
    network.connect(nb.Connection(source=0, target=1, weight=20.0, max_weight=20.0, delay=1))
    network.connect(
        nb.Connection(source=1, target=2, weight=1.0, max_weight=5.0, delay=1, synapse_rule=0)
    )
    network.connect(nb.Connection(source=0, target=3, weight=1.0, max_weight=1.0, delay=1))
    network.connect(
        nb.Connection(source=3, target=2, weight=0.0, max_weight=0.0, delay=1, trigger_rule=0)
    )
    return network, 1


def test_incremental_builder_writes_valid_file(tmp_path: Path):
    path = tmp_path / "network.nbnet"
    with nb.NbnetDescriptionBuilder(path) as builder:
        input_layer = builder.add_layer(nb.NeuronLayer.input_spike(3))
        output_layer = builder.add_layer(_lif_layer())
        assert input_layer.first_neuron_id == 0
        assert output_layer.first_neuron_id == 3
        builder.append_connections(
            [0, 1, 2],
            [3, 3, 3],
            weight=[1.0, 2.0, 3.0],
            max_weight=4.0,
            delay=1,
        )

    assert builder.result is not None
    assert builder.result.connection_count == 3
    assert builder.result.neuron_count == 4
    info = nb.nbnet.inspect(path, verify_checksum=True)
    assert info.connection_count == 3
    assert info.neuron_count == 4
    report = nb.nbnet.validate(path)
    assert report.valid
    assert report.errors == ()


def test_from_batches_single_append_abort_and_is_valid(tmp_path: Path):
    path = tmp_path / "batched.nbnet"
    result = nb.nbnet.from_batches(
        path,
        [nb.NeuronLayer.input_spike(2), _lif_layer()],
        [
            {"source": [0], "target": [2], "weight": [1.25], "delay": 1},
            {"source": [1], "target": [2], "weight": [2.5], "delay": 2},
        ],
    )
    assert result.connection_count == 2
    assert nb.nbnet.is_valid(path)

    single_path = tmp_path / "single.nbnet"
    builder = nb.NbnetDescriptionBuilder(single_path)
    builder.add_layer(nb.NeuronLayer.input_spike(1))
    builder.add_layer(_lif_layer())
    assert builder.append_connection(nb.Connection(0, 1, weight=3.0, delay=1)) == 1
    first = builder.finalize()
    assert builder.finalize() is first
    assert nb.nbnet.is_valid(single_path)
    with pytest.raises(RuntimeError, match="closed"):
        builder.add_layer(nb.NeuronLayer.input_spike(1))

    aborted_path = tmp_path / "aborted.nbnet"
    aborted = nb.NbnetDescriptionBuilder(aborted_path)
    aborted.add_layer(nb.NeuronLayer.input_spike(1))
    aborted.abort()
    aborted.abort()
    assert not aborted_path.exists()
    assert not list(tmp_path.glob(".aborted.nbnet.*.records.tmp"))
    with pytest.raises(RuntimeError, match="closed"):
        aborted.append_connections([0], [0])

    corrupted = tmp_path / "corrupted.nbnet"
    corrupted.write_bytes(path.read_bytes()[:-1])
    assert not nb.nbnet.is_valid(corrupted)


def test_from_batches_aborts_cleanly_when_batch_generation_fails(tmp_path: Path):
    path = tmp_path / "failed.nbnet"

    def batches():
        yield {"source": [0], "target": [1], "weight": 1.0}
        raise RuntimeError("batch producer failed")

    with pytest.raises(RuntimeError, match="batch producer failed"):
        nb.nbnet.from_batches(
            path,
            [nb.NeuronLayer.input_spike(1), _lif_layer()],
            batches(),
        )
    assert not path.exists()
    assert not list(tmp_path.glob(".failed.nbnet.*.records.tmp"))


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
@pytest.mark.parametrize("dense", [False, True])
def test_additive_kernel_learning_fast_and_streaming(tmp_path: Path, dense: bool):
    network, plastic_connection = _additive_kernel_network(dense=dense)
    path = tmp_path / ("additive_dense.nbnet" if dense else "additive_main.nbnet")
    nb.nbnet.from_network(network, path)
    config = nb.SimulationConfig(steps=24, timestep=1.0, event_queue="timing_wheel", timing_wheel_size=64)

    results = []
    for source in (network, path):
        sim = nb.Simulation(source, config).init()
        before = sim.get_connection_weight(plastic_connection)
        if dense:
            sim.add_external_spikes([1, 4, 8, 12], [0, 0, 0, 0])
        else:
            sim.add_external_spikes([1, 4, 8, 11, 15, 18], [0, 1, 0, 1, 0, 1])
        sim.run()
        after = sim.get_connection_weight(plastic_connection)
        assert after != pytest.approx(before)
        results.append(after)

    assert results[1] == pytest.approx(results[0], abs=1e-6)


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_unknown_learning_rule_raises_instead_of_dereferencing_null():
    network = _network()
    network.add_learning_rule(nb.LearningRule("DefinitelyUnknownLearningRule"))
    with pytest.raises(RuntimeError, match="Failed to construct learning rule"):
        nb.Simulation(network, nb.SimulationConfig(steps=2, timestep=0.1))


def test_from_network_detects_corruption(tmp_path: Path):
    path = tmp_path / "network.nbnet"
    result = nb.nbnet.from_network(_network(), path)
    assert result.connection_count == 1

    content = bytearray(path.read_bytes())
    content[-1] ^= 0x01
    path.write_bytes(content)
    report = nb.nbnet.validate(path)
    assert not report.valid
    assert "checksum mismatch" in report.errors[0]

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        nb.Simulation(path, nb.SimulationConfig(steps=2, timestep=0.1))


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_cpp_loader_rejects_bad_header_and_truncation(tmp_path: Path):
    source = tmp_path / "source.nbnet"
    nb.nbnet.from_network(_network(), source)
    config = nb.SimulationConfig(steps=2, timestep=0.1)

    bad_magic = tmp_path / "bad_magic.nbnet"
    content = bytearray(source.read_bytes())
    content[0] ^= 0x01
    bad_magic.write_bytes(content)
    with pytest.raises(RuntimeError, match="invalid nbnet magic"):
        nb.Simulation(bad_magic, config)

    truncated = tmp_path / "truncated.nbnet"
    truncated.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match="file size mismatch"):
        nb.Simulation(truncated, config)


def test_builder_refuses_implicit_overwrite(tmp_path: Path):
    path = tmp_path / "network.nbnet"
    nb.nbnet.from_network(_network(), path)
    with pytest.raises(FileExistsError):
        nb.nbnet.from_network(_network(), path)


def test_builder_rejects_out_of_range_neuron_id(tmp_path: Path):
    with nb.NbnetDescriptionBuilder(tmp_path / "invalid.nbnet") as builder:
        builder.add_layer(nb.NeuronLayer.input_spike(1))
        with pytest.raises(ValueError, match="out of range"):
            builder.append_connections([0], [1])


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_streaming_and_in_memory_simulations_are_equivalent(tmp_path: Path):
    network = _network()
    path = tmp_path / "network.nbnet"
    nb.nbnet.from_network(network, path)
    config = nb.SimulationConfig(steps=20, timestep=0.1)

    fast = nb.Simulation(network, config).init()
    streaming = nb.Simulation(
        path,
        config,
        build_options=nb.StreamingBuildOptions(memory_budget_mb=1, mmap=True, verify_checksum=True),
    ).init()
    fast.add_external_spikes([1], [0]).run()
    streaming.add_external_spikes([1], [0]).run()

    assert streaming.build_stats["mode"] == "streaming_file_cpp"
    assert streaming.build_stats["runtime_build_path"] == "streaming_direct"
    assert streaming.build_stats["connection_count"] == 1
    assert streaming.output_spikes() == fast.output_spikes()
    assert streaming.get_connection_weight(0) == pytest.approx(fast.get_connection_weight(0), abs=1e-7)


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_weight_loader_rejects_malformed_and_mismatched_snapshots_without_mutation(tmp_path: Path):
    sim = nb.Simulation(_network(), nb.SimulationConfig(steps=4, timestep=0.1)).init()
    sim.set_connection_weight(0, 4.5)

    cases = {
        "bad_header": "not-a-weight-file\n",
        "truncated": (
            "NPGR_WEIGHT_SNAPSHOT 1\n"
            "time_step 0\n"
            "block main 0 1\n"
            "rle\n"
            "1 2.0\n"
        ),
        "wrong_count": (
            "NPGR_WEIGHT_SNAPSHOT 1\n"
            "time_step 0\n"
            "block main 0 2\n"
            "rle\n"
            "2 2.0\n"
            "end_block\n"
            "end\n"
        ),
        "nonfinite": (
            "NPGR_WEIGHT_SNAPSHOT 1\n"
            "time_step 0\n"
            "block main 0 1\n"
            "rle\n"
            "1 nan\n"
            "end_block\n"
            "end\n"
        ),
    }
    for name, content in cases.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(RuntimeError):
            sim.load_weights(path)
        assert sim.get_connection_weight(0) == pytest.approx(4.5)


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
@pytest.mark.parametrize("streaming", [False, True])
def test_reset_preserves_learned_weight_with_debug_monitor(tmp_path: Path, streaming: bool):
    network, plastic_connection = _additive_kernel_network(dense=False)
    source = network
    if streaming:
        source = tmp_path / "reset_learning.nbnet"
        nb.nbnet.from_network(network, source)
    monitor_dir = tmp_path / ("streaming_monitor" if streaming else "fast_monitor")
    sim = nb.Simulation(source, nb.SimulationConfig(steps=24, timestep=1.0)).init()
    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=monitor_dir,
            all_neurons=True,
            record_state=True,
            record_spikes=True,
            record_weights=True,
            flush_interval_steps=4,
        )
    )
    sim.add_external_spikes([1, 4, 8, 11, 15, 18], [0, 1, 0, 1, 0, 1]).run()
    learned = sim.get_connection_weight(plastic_connection)
    assert learned != pytest.approx(8.0)
    sim.reset()
    assert sim.get_connection_weight(plastic_connection) == pytest.approx(learned)
    sim.add_external_spikes([1, 4, 8], [0, 1, 0]).run(12)
    sim.flush().disable_debug_monitor()
    assert (monitor_dir / "weights.csv").stat().st_size > 0


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_streaming_dense_cross_subnetwork_uses_direct_path(tmp_path: Path):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.lif_double(1, dense_name="dense_a"))
    network.add_layer(nb.NeuronLayer.lif_double(1, dense_name="dense_b"))
    network.connect(nb.Connection(0, 1, weight=7.0, max_weight=7.0, delay=2))
    path = tmp_path / "dense_cross.nbnet"
    nb.nbnet.from_network(network, path)

    sim = nb.Simulation(path, nb.SimulationConfig(steps=4, timestep=1.0)).init()
    assert sim.build_stats["runtime_build_path"] == "streaming_direct"
    assert sim.dense_subnetwork_count == 2
    assert sim.dense_subnetwork_name(0) == "dense_a"
    assert sim.dense_subnetwork_name(1) == "dense_b"
    assert sim.get_connection_weight(0) == pytest.approx(7.0, abs=1e-7)


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_streaming_mixed_dense_routes_preserve_original_weight_indices(tmp_path: Path):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))                 # 0 main
    network.add_layer(nb.NeuronLayer.lif_double(2, dense_name="a")) # 1..2 dense a
    network.add_layer(nb.NeuronLayer.lif_double(1, dense_name="b")) # 3 dense b
    network.add_layer(_lif_layer())                                  # 4 main
    connections = [
        nb.Connection(0, 4, weight=1.0, max_weight=5.0, delay=1),  # main -> main
        nb.Connection(0, 1, weight=2.0, max_weight=5.0, delay=1),  # main -> dense
        nb.Connection(1, 2, weight=3.0, max_weight=5.0, delay=1),  # dense internal
        nb.Connection(2, 3, weight=4.0, max_weight=5.0, delay=2),  # dense -> dense
        nb.Connection(3, 4, weight=5.0, max_weight=5.0, delay=1),  # dense -> main
    ]
    for connection in connections:
        network.connect(connection)
    path = tmp_path / "mixed_routes.nbnet"
    nb.nbnet.from_network(network, path)
    config = nb.SimulationConfig(steps=8, timestep=1.0)

    fast = nb.Simulation(network, config).init()
    streaming = nb.Simulation(path, config).init()
    assert streaming.build_stats["runtime_build_path"] == "streaming_direct"
    assert streaming.dense_subnetwork_count == fast.dense_subnetwork_count == 2
    for index, expected in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        assert fast.get_connection_weight(index) == pytest.approx(expected, abs=1e-7)
        assert streaming.get_connection_weight(index) == pytest.approx(expected, abs=1e-7)


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_streaming_outer_dynamic_and_inputconv_metadata(tmp_path: Path):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_outer_dynamic(nb.OuterDynamic.spike_counter(name="counter", slot_count=2, type_count=2))
    network.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=0,
            target_outer_dynamic=0,
            target_joint=1,
            synapse_type=0,
            weight=2.5,
            delay=1,
        )
    )
    network.add_input_conv(nb.InputConv.v1_bar(width=2, height=2, channels=1))
    path = tmp_path / "advanced.nbnet"
    nb.nbnet.from_network(network, path)

    sim = nb.Simulation(path, nb.SimulationConfig(steps=4, timestep=1.0)).init()
    assert sim.build_stats["runtime_build_path"] == "streaming_direct"
    assert sim.input_conv_count == 1
    assert sim.input_conv_output_count(0) > 0
    sim.add_external_spikes([0], [0]).run(3)
    snapshot = sim.outer_dynamic_spike_counter_snapshot("counter")
    assert snapshot["spike_counts"][1] == 1
    assert snapshot["weighted_sums"][1] == pytest.approx(2.5)


@pytest.mark.skipif(not nb.backend_info()["native_extension_loaded"], reason="native extension is not built")
def test_streaming_learning_rule_and_original_weight_index(tmp_path: Path):
    network = _network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_learning_rule(
        nb.LearningRule(
            "CerebullarLearningRule",
            parameters={"fixweightchange": nb.float32(0.2), "kernalchange": nb.float32(-0.3)},
        )
    )
    network.connect(nb.Connection(source=2, target=1, weight=0.0, max_weight=3.0, delay=1, trigger_rule=0))
    path = tmp_path / "learning.nbnet"
    nb.nbnet.from_network(network, path)
    config = nb.SimulationConfig(steps=8, timestep=0.1)

    fast = nb.Simulation(network, config).init()
    streaming = nb.Simulation(path, config).init()
    assert streaming.build_stats["runtime_build_path"] == "streaming_direct"
    for index in range(2):
        assert streaming.get_connection_weight(index) == pytest.approx(
            fast.get_connection_weight(index), abs=1e-7
        )
    streaming.set_connection_weight(1, 1.25)
    assert streaming.get_connection_weight(1) == pytest.approx(1.25, abs=1e-7)
