from pathlib import Path
import importlib
import sys
import time
from array import array

import pytest

import neuronbridge as nb

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "python"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _dense_baseline_builders import (
    build_dense_mixed_current_network,
    build_dense_mixed_network,
    build_dense_run_no_debug_network,
)
from _ei_baseline_builders import (
    NEURON_COUNT,
    build_main_ei_program,
    load_connectivity,
)
from _handwriting_stage1 import ArmParameters
from _handwriting_stage1 import compute_stroke as compute_handwriting_stage1_stroke
from _handwriting_stage1 import load_stroke_path_from_directory
from _handwriting_stage1 import mean_squared_error
from _handwriting_stage2 import Stage2Config
from _handwriting_stage2 import decode_population_spikes
from _handwriting_stage2 import find_existing_stage2_input
from _handwriting_stage2 import load_binary_spike_series
from _handwriting_stage2 import load_binary_weight_windows
from _handwriting_stage2 import simulate_cm_forward_exact
from _handwriting_stage3 import Stage3Data
from _handwriting_stage3 import Stage3Config as Stage3RuntimeConfig
from _handwriting_stage3 import MatrixF32
from _handwriting_stage3 import Stage3Mode
from _handwriting_stage3 import build_padded_path
from _handwriting_stage3 import parse_stage3_mode
from _handwriting_stage3 import run_stage3_numpy
from _handwriting_stage3 import target_path_for_mode
from _handwriting_stage2 import BinarySpikeSeries, BinaryWeightWindows, Shape2D, Shape3D
from handwriting_stage3_framework import run_framework as run_stage3_framework
from handwriting_stage4_different_position import run_stage4
from my_first_app import build_network as build_my_first_app_network
from my_first_app import run_example as run_my_first_app
from _pattern_motion_builder import (
    PatternMotionSpec,
    build_pattern_motion_network,
    write_stimulus_file,
)
from cerebellum_framework import CerebellumConfig
from cerebellum_framework import run_example as run_cerebellum_framework
from zmqcommunication_client import SpikeRecord
from zmqcommunication_client import pack_spike_batch
from zmqcommunication_client import run_example as run_zmqcommunication_client
from zmqcommunication_client import unpack_spike_batch
from pattern_motion import run_pattern_motion


def _external_data_dir(name: str) -> Path:
    from _example_paths import data_dir

    path = data_dir(name)
    if not path.is_dir():
        pytest.skip(
            f"optional example data is not installed: {path}; "
            "set NEURONBRIDGE_DATA_ROOT to an extracted data bundle"
        )
    return path


@pytest.mark.parametrize(
    "module_name",
    [path.stem for path in sorted(EXAMPLES_DIR.glob("*.py"))],
)
def test_all_migrated_example_modules_import(module_name: str):
    importlib.import_module(module_name)


def _all_to_all(source_begin, source_count, target_begin, target_count, *, weight, delay=1, synapse_type=0):
    return nb.Connection(
        source=[
            source_begin + source
            for source in range(source_count)
            for _target in range(target_count)
        ],
        target=[
            target_begin + target
            for _source in range(source_count)
            for target in range(target_count)
        ],
        weight=[float(weight)] * (source_count * target_count),
        max_weight=[10.0] * (source_count * target_count),
        delay=[int(delay)] * (source_count * target_count),
        synapse_type=[int(synapse_type)] * (source_count * target_count),
    )


def test_backend_info_scaffold():
    info = nb.backend_info()
    assert info["package"] == "neuronbridge"


def test_open_existing_debug_monitor():
    root = Path(__file__).resolve().parents[2]
    result_dir = root / "dense_debug_monitor_real_out"
    if not result_dir.exists():
        return
    result = nb.open_debug_monitor(result_dir)
    assert isinstance(result.file_sizes(), dict)


def _write_monitor_csv(path: Path, name: str, contents: str) -> None:
    (path / name).write_text(contents, encoding="utf-8", newline="\n")


def _plot_pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def test_debug_monitor_weight_plotting(tmp_path):
    plt = _plot_pyplot()
    _write_monitor_csv(
        tmp_path,
        "weights.csv",
        "time_step,component_kind,component_index,component_name,synapse_id,value\n"
        "0,MainNetwork,0,main_network,0,1.0\n"
        "0,MainNetwork,0,main_network,1,2.0\n"
        "0,DenseSubnetwork,0,dense_a,0,9.0\n"
        "1,MainNetwork,0,main_network,0,2.0\n"
        "1,MainNetwork,0,main_network,1,4.0\n"
        "1,DenseSubnetwork,0,dense_a,0,8.0\n",
    )
    result = nb.open_debug_monitor(tmp_path)

    trace = result.plot_weight_trace("main_network", [0, 1])
    assert len(trace.lines) == 2
    assert list(trace.lines[0].get_ydata()) == [1.0, 2.0]
    assert trace.lines[0].get_label() == "main_network:0"

    summary = result.plot_weight_summary("main_network")
    mean_line = next(line for line in summary.lines if line.get_label() == "mean")
    assert list(mean_line.get_ydata()) == [1.5, 3.0]
    assert len(summary.collections) == 1

    distribution = result.plot_weight_distribution("main_network", time_step=1, bins=2)
    assert len(distribution.patches) == 2
    with pytest.raises(ValueError, match="synapse_ids"):
        result.plot_weight_trace("main_network", [99])
    plt.close("all")


def test_debug_monitor_outer_dynamic_plotting(tmp_path):
    plt = _plot_pyplot()
    _write_monitor_csv(
        tmp_path,
        "outer_dynamic_state.csv",
        "time_step,field_name,index,value\n"
        "0,q,0,1.0\n0,q,1,2.0\n0,q_des,0,1.5\n0,q_des,1,1.5\n0,qv,0,0.1\n0,qv,1,0.2\n0,tau_total,0,4.0\n0,tau_total,1,5.0\n"
        "1,q,0,2.0\n1,q,1,3.0\n1,q_des,0,1.5\n1,q_des,1,2.5\n1,qv,0,0.3\n1,qv,1,0.4\n1,tau_total,0,6.0\n1,tau_total,1,7.0\n",
    )
    result = nb.open_debug_monitor(tmp_path)

    trace = result.plot_outer_dynamic_trace()
    assert len(trace.lines) == 4
    stacked = result.plot_outer_dynamic_trace(layout="stacked")
    assert len(stacked) == 2

    error = result.plot_outer_dynamic_tracking_error(indices=0)
    assert list(error.lines[0].get_ydata()) == [-0.5, 0.5]
    phase = result.plot_outer_dynamic_phase_plane()
    assert list(phase.lines[0].get_xdata()) == [1.0, 2.0]
    assert list(phase.lines[0].get_ydata()) == [0.1, 0.3]
    torque = result.plot_outer_dynamic_torque()
    assert torque.get_ylabel() == "total torque"
    with pytest.raises(ValueError, match="qdd"):
        result.plot_outer_dynamic_trace(fields=("qdd",))

    multi_dir = tmp_path / "multi_outer_dynamic"
    multi_dir.mkdir()
    _write_monitor_csv(
        multi_dir,
        "outer_dynamic_state.csv",
        "time_step,component_index,component_name,field_name,index,value\n"
        "0,0,arm_a,q,0,1.0\n0,1,arm_b,q,0,10.0\n"
        "1,0,arm_a,q,0,2.0\n1,1,arm_b,q,0,20.0\n"
        "0,0,arm_a,qv,0,0.1\n0,1,arm_b,qv,0,1.0\n"
        "1,0,arm_a,qv,0,0.2\n1,1,arm_b,qv,0,2.0\n",
    )
    multi_result = nb.open_debug_monitor(multi_dir)
    selected = multi_result.plot_outer_dynamic_trace(fields=("q",), indices=0, outer_dynamic="arm_b")
    assert list(selected.lines[0].get_ydata()) == [10.0, 20.0]
    selected_by_index = multi_result.plot_outer_dynamic_phase_plane(outer_dynamic=1)
    assert list(selected_by_index.lines[0].get_ydata()) == [1.0, 2.0]
    plt.close("all")


def test_native_multi_outer_dynamic_monitor_schema(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_outer_dynamic(nb.OuterDynamic.strict_matlab_planar_arm_2dof(name="arm_a"))
    network.add_outer_dynamic(nb.OuterDynamic.strict_matlab_planar_arm_2dof(name="arm_b"))
    monitor_dir = tmp_path / "multi_outer_dynamic_native"
    sim = nb.Simulation(network, nb.SimulationConfig(steps=8, timestep=1.0))
    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=monitor_dir,
            record_spikes=False,
            record_state=False,
            record_pending_channels=False,
            record_outer_dynamic_state=True,
        )
    )
    sim.run(4)
    sim.flush()

    rows = list(sim.result().iter_rows("outer_dynamic_state"))
    assert {row["component_name"] for row in rows} == {"arm_a", "arm_b"}
    assert {row["component_index"] for row in rows} == {"0", "1"}
    states = sim.outer_dynamic_states()
    assert {state["component_name"] for state in states} == {"arm_a", "arm_b"}
    plt = _plot_pyplot()
    arm_b = sim.result().plot_outer_dynamic_trace(fields=("q",), indices=0, outer_dynamic="arm_b")
    assert len(arm_b.lines[0].get_xdata()) > 0
    plt.close("all")


def test_python_network_description_to_dict():
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_double(2, dense_name="dense_py", V_th=-64.5, dense_steps_to_keep=16))
    network.connect(nb.Connection(0, [1, 2], weight=[1.0, 2.0], max_weight=10.0, delay=1))
    data = network.to_dict()
    assert network.neuron_count == 3
    assert data["layers"][1]["parameters"]["dense_subnetwork_name"] == "dense_py"
    assert data["connections"][0]["target"] == [1, 2]
    if nb.backend_info()["native_extension_loaded"]:
        native = network.to_native()
        assert native.layer_count == 2
        assert native.neuron_count == 3


def test_outer_dynamic_description_to_dict():
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2))
    network.add_outer_dynamic(nb.OuterDynamic.spike_counter(name="counter_sink", slot_count=3, type_count=2))
    network.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=[0, 1],
            target_outer_dynamic=0,
            target_joint=[1, 1],
            synapse_type=[0, 1],
            weight=[2.0, 3.5],
            delay=1,
        )
    )
    data = network.to_dict()
    assert data["outer_dynamics"][0]["model"] == "OuterDynamicSpikeCounter"
    assert data["outer_dynamics"][0]["parameters"]["slot_count"] == 3
    assert data["outer_dynamic_connections"][0]["target_joint"] == [1, 1]
    if nb.backend_info()["native_extension_loaded"]:
        native = network.to_native()
        assert native.outer_dynamic_count == 1
        assert native.outer_dynamic_connection_block_count == 1


def test_input_conv_and_typed_parameters_to_dict():
    network = nb.Network()
    network.add_input_conv(
        nb.InputConv.v1_bar(
            width=8,
            height=8,
            parameters={"speed": nb.float32(1.5), "link_lengths": nb.float64_list([0.3, 0.25])},
            output_target="dense_subnetwork",
            target_dense_subnetwork_name="dense_py",
            output_source_indices=[0, 1],
            output_target_neuron_ids=[10, 11],
            output_scales=[1.0, 0.5],
        )
    )
    data = network.to_dict()
    assert data["input_convs"][0]["model"] == "InputConvV1"
    assert data["input_convs"][0]["output_target"] == "dense_subnetwork"
    assert data["input_convs"][0]["parameters"]["link_lengths"] == [0.3, 0.25]
    if nb.backend_info()["native_extension_loaded"]:
        native_data = network.to_native().to_dict()
        assert native_data["input_convs"][0]["parameters"]["link_lengths"] == [0.3, 0.25]
        assert native_data["input_convs"][0]["output_target"] == "dense_subnetwork"


def test_planar_arm_2dof_convenience_types():
    network = nb.Network()
    network.add_outer_dynamic(
        nb.OuterDynamic.planar_arm_2dof(
            name="arm",
            link_lengths=[0.4, 0.3],
            link_masses=[1.2, 0.8],
            joint_damping=[0.05, 0.04],
            pd_kp=[14.0, 11.0],
            pd_kd=[2.5, 2.0],
            trajectory_frequency_hz=0.35,
            dcn_torque_gain=0.0015,
            spike_retention_steps=512,
            state_feedback_single=nb.FeedbackSingleEncoding(
                bins=[7, 7, 7, 7],
                neuron_indices_by_joint_variable=[[[0], [1], [2], [3]], [[4], [5], [6], [7]]],
            ),
        )
    )
    data = network.to_dict()
    assert data["outer_dynamics"][0]["parameters"]["link_lengths"] == [0.4, 0.3]
    assert data["outer_dynamics"][0]["state_feedback_single"]["bins"] == [7, 7, 7, 7]
    if nb.backend_info()["native_extension_loaded"]:
        native = network.to_native()
        native_data = native.to_dict()
        assert native.outer_dynamic_count == 1
        assert native_data["outer_dynamics"][0]["parameters"]["trajectory_frequency_hz"] == 0.35


def test_input_conv_v1_file_and_plaid_helpers():
    file_conv = nb.InputConv.v1_file(
        stimulus_file_path="stimulus.bin",
        width=32,
        height=32,
        channels=1,
        frame_hold_steps=2,
    )
    plaid_conv = nb.InputConv.v1_plaid(width=16, height=16, plaid_direction_b_deg=90.0)
    assert file_conv.to_dict()["parameters"]["stimulus_mode"] == "file"
    assert file_conv.to_dict()["parameters"]["frame_hold_steps"] == 2
    assert plaid_conv.to_dict()["parameters"]["stimulus_mode"] == "plaid"
    assert plaid_conv.to_dict()["parameters"]["plaid_direction_b_deg"] == 90.0


def test_input_conv_dynamic_frame_protocol_helpers():
    frame = nb.InputConvFrame(
        time_step=7,
        source_camera_index=2,
        width=2,
        height=2,
        channels=1,
        pixel_format=nb.InputConvPixelFormat.UINT8_GRAY,
        bytes=bytes([1, 2, 3, 4]),
    )
    response = nb.pack_input_conv_frame(frame)
    decoded = nb.unpack_input_conv_frame(response)
    assert decoded.time_step == 7
    assert decoded.source_camera_index == 2
    assert decoded.width == 2
    assert decoded.height == 2
    assert decoded.channels == 1
    assert decoded.pixel_format == nb.InputConvPixelFormat.UINT8_GRAY
    assert decoded.bytes == bytes([1, 2, 3, 4])

    async_payload = nb.pack_input_conv_frame(frame, async_message=True)
    async_decoded = nb.unpack_input_conv_frame(async_payload, async_message=True)
    assert async_decoded == frame

    request = nb.InputConvFrameRequest(
        time_step=9,
        inputconv_index=1,
        source_camera_index=2,
        width=8,
        height=4,
        channels=3,
        preferred_pixel_format="uint8_rgb",
        max_lag_steps=5,
    )
    request_decoded = nb.unpack_input_conv_frame_request(nb.pack_input_conv_frame_request(request))
    assert request_decoded.time_step == 9
    assert request_decoded.inputconv_index == 1
    assert request_decoded.source_camera_index == 2
    assert request_decoded.preferred_pixel_format == nb.InputConvPixelFormat.UINT8_RGB
    assert request_decoded.max_lag_steps == 5

    with pytest.raises(ValueError, match="payload"):
        nb.InputConvFrame(
            time_step=0,
            source_camera_index=0,
            width=2,
            height=2,
            channels=1,
            bytes=b"\x00",
        )
    with pytest.raises(ValueError, match="invalid InputConv frame request magic"):
        bad_request = bytearray(nb.pack_input_conv_frame_request(request))
        bad_request[:4] = (0).to_bytes(4, "little")
        nb.unpack_input_conv_frame_request(bytes(bad_request))
    with pytest.raises(ValueError, match="invalid InputConv frame response magic"):
        bad_response = bytearray(response)
        bad_response[:4] = (0).to_bytes(4, "little")
        nb.unpack_input_conv_frame(bytes(bad_response))
    with pytest.raises(ValueError, match="invalid async InputConv frame header"):
        bad_async = bytearray(async_payload)
        bad_async[:4] = (0).to_bytes(4, "little")
        nb.unpack_input_conv_frame(bytes(bad_async), async_message=True)
    with pytest.raises(ValueError, match="incomplete InputConv frame payload"):
        nb.unpack_input_conv_frame(response[:-2])


def test_input_conv_frame_publisher_summary():
    pytest.importorskip("zmq")

    frame = nb.InputConvFrame(
        time_step=3,
        source_camera_index=1,
        width=2,
        height=2,
        channels=1,
        bytes=bytes([1, 2, 3, 4]),
    )
    with nb.InputConvFramePublisher(topic="summary_topic") as publisher:
        assert publisher.port > 0
        assert publisher.topic == "summary_topic"
        publisher.publish_frame(frame, repeat=2)
        summary = publisher.summary()
        assert summary["published_frames"] == 1
        assert summary["published_messages"] == 2
        assert summary["last_time_step"] == 3
        assert summary["last_source_camera_index"] == 1
    assert publisher.closed


def test_input_conv_frame_client_server_roundtrip():
    pytest.importorskip("zmq")

    def provide_frame(request: nb.InputConvFrameRequest) -> nb.InputConvFrame:
        return nb.InputConvFrame(
            time_step=request.time_step,
            source_camera_index=request.source_camera_index,
            width=request.width,
            height=request.height,
            channels=request.channels,
            pixel_format=request.preferred_pixel_format,
            bytes=bytes([12, 13, 14, 15]),
        )

    request = nb.InputConvFrameRequest(
        time_step=5,
        inputconv_index=2,
        source_camera_index=1,
        width=2,
        height=2,
        channels=1,
        preferred_pixel_format="uint8_gray",
        max_lag_steps=3,
    )
    with nb.InputConvFrameServer(provide_frame) as server:
        with nb.InputConvFrameClient(port=server.port) as client:
            frame = client.request_frame(request)
            client_summary = client.summary()
        server_summary = server.summary()

    assert frame.time_step == 5
    assert frame.source_camera_index == 1
    assert frame.bytes == bytes([12, 13, 14, 15])
    assert client_summary["requested_frames"] == 1
    assert client_summary["received_frames"] == 1
    assert client_summary["failed_requests"] == 0
    assert server_summary["received_requests"] == 1
    assert server_summary["sent_frames"] == 1


def test_phase2_validation_errors():
    with pytest.raises(ValueError, match="target_dense_subnetwork_name"):
        nb.InputConv.v1_bar(output_target="dense_subnetwork")
    with pytest.raises(ValueError, match="same length"):
        nb.InputConv(output_source_indices=[0], output_target_neuron_ids=[1, 2])
    with pytest.raises(ValueError, match="unsupported native parameter kind"):
        nb.NativeParameter("bad_kind", 1)
    with pytest.raises(TypeError, match="mixed parameter list"):
        nb.NeuronLayer.input_spike(1, bad=[1, 2.0])


def test_phase3_simulation_lifecycle_and_weight_io(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(1))
    network.connect(nb.Connection(0, 1, weight=1.25, max_weight=10.0, delay=1))

    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=0.1))
    assert not sim.initialized
    sim.init()
    assert sim.initialized
    assert sim.get_connection_weight(0) == pytest.approx(1.25)
    sim.set_connection_weight(0, 2.5)
    assert sim.get_connection_weight(0) == pytest.approx(2.5)
    sim.add_external_spikes([0], [0])
    sim.run(1)

    monitor_dir = tmp_path / "monitor"
    sim.enable_debug_monitor(nb.DebugMonitorConfig(output_dir=monitor_dir))
    sim.run(1)
    sim.flush()
    assert sim.result().path == monitor_dir


def test_baseline_simulation_weight_smoke(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(1))
    network.add_layer(nb.NeuronLayer.lif_double(2, dense_name="weight_io", dense_steps_to_keep=16, V_th=nb.float32(-64.5)))
    network.connect(nb.Connection(0, 1, weight=1.25, max_weight=10.0, delay=1))
    network.connect(nb.Connection(2, 3, weight=2.50, max_weight=10.0, delay=1))
    network.connect(nb.Connection(0, 2, weight=3.75, max_weight=10.0, delay=1))

    sim = nb.Simulation(network, nb.SimulationConfig(steps=8, timestep=0.1))
    sim.init()
    assert sim.get_connection_weight(0) == pytest.approx(1.25)
    assert sim.get_connection_weight(1) == pytest.approx(2.50)
    assert sim.get_connection_weight(2) == pytest.approx(3.75)

    snapshot = tmp_path / "simulation_weight_smoke_snapshot.txt"
    sim.save_weights(snapshot)
    sim.set_connection_weight(0, 7.25)
    sim.set_connection_weight(1, 8.50)
    with pytest.raises(RuntimeError, match="SetConnectionWeight failed"):
        sim.set_connection_weight(2, 9.75)

    assert sim.get_connection_weight(0) == pytest.approx(7.25)
    assert sim.get_connection_weight(1) == pytest.approx(8.50)
    assert sim.get_connection_weight(2) == pytest.approx(3.75)

    sim.load_weights(snapshot)
    assert sim.get_connection_weight(0) == pytest.approx(1.25)
    assert sim.get_connection_weight(1) == pytest.approx(2.50)
    assert sim.get_connection_weight(2) == pytest.approx(3.75)


def test_my_first_app_python_migration():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = build_my_first_app_network()
    assert network.neuron_count == 8
    assert len(network.connections) == 1
    assert len(network.connections[0].normalized()["source"]) == 16

    sim = run_my_first_app(total_steps=100, dt_ms=0.1)
    assert sim.initialized
    assert isinstance(sim.output_spikes(), list)
    assert sim.neuron_state(4)["original_neuron_id"] == 4


def test_handwriting_stage1_python_migration():
    input_dir = _external_data_dir("handwriting_stage1")
    path = load_stroke_path_from_directory(input_dir, "1")
    result = compute_handwriting_stage1_stroke(path, ArmParameters(), 0.01)

    assert len(result.desired_path.x) == 400
    assert len(result.joints.theta1) == 400
    assert len(result.torques.q1) == 400
    assert len(result.replay.hand_path.x) == 400
    assert mean_squared_error(result.desired_path.x, result.replay.hand_path.x) == pytest.approx(5.59612e-08, abs=1e-13)
    assert mean_squared_error(result.desired_path.y, result.replay.hand_path.y) == pytest.approx(4.00717e-08, abs=1e-13)


def test_handwriting_stage2_python_migration_smoke():
    input_dir = _external_data_dir("handwriting_stage2")
    cfg = Stage2Config()
    ext_spikes = load_binary_spike_series(
        find_existing_stage2_input(input_dir, "sspk_ext", ".bin"),
        find_existing_stage2_input(input_dir, "sspk_ext", ".shape.txt"),
    )
    weights = load_binary_weight_windows(
        find_existing_stage2_input(input_dir, "ww_extCM1", ".bin"),
        find_existing_stage2_input(input_dir, "ww_extCM1", ".shape.txt"),
        max_windows=5,
    )
    result = simulate_cm_forward_exact(cfg, ext_spikes, weights, max_windows=5)
    torque = decode_population_spikes(result.population_counts, cfg.n_pop)

    assert result.simulated_windows == 5
    assert len(result.population_counts) == 40
    assert len(torque.q1) == 5
    assert sum(result.population_counts) >= 0


def test_handwriting_stage4_python_migration(tmp_path):
    input_dir = _external_data_dir("handwriting_stage4_different_position")
    results = run_stage4(input_dir, tmp_path / "stage4", stem="1")

    assert len(results) == 9
    assert results[0].dx == pytest.approx(-0.1)
    assert results[0].dy == pytest.approx(0.1)
    assert results[0].mse_x == pytest.approx(1.5043940083149732e-05, abs=1e-12)
    assert results[4].mse_y == pytest.approx(1.94593802096289e-06, abs=1e-12)


def test_handwriting_stage3_python_migration_scaffold():
    assert parse_stage3_mode("test") is Stage3Mode.TEST
    assert parse_stage3_mode("testD") is Stage3Mode.TEST_D
    assert parse_stage3_mode("train_g") is Stage3Mode.TRAIN_G

    first = type("Dummy", (), {})()
    first.x = [1.0, 2.0]
    first.y = [3.0, 4.0]
    second = type("Dummy", (), {})()
    second.x = [5.0]
    second.y = [6.0]
    padded = build_padded_path(first, second, hold=2)
    assert padded.x == [1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0]
    assert padded.y == [3.0, 3.0, 3.0, 4.0, 6.0, 6.0, 6.0]

    data = Stage3Data(
        sspk_ext=None,
        sspk_ext_e=None,
        sspk_ext_i=None,
        sspk_ext_mm=None,
        sspk_sig=None,
        sspk_sm=None,
        ww_extcm1=None,
        ww_extcm2=None,
        ww_extcm3=None,
        a1_flat=[],
        a2_flat=[],
        a3_flat=[],
        x1y1=first,
        x2y2=second,
        x3y3=type("Dummy", (), {"x": [7.0], "y": [8.0]})(),
    )
    assert target_path_for_mode(data, Stage3Mode.TEST).x[-1] == 5.0
    assert target_path_for_mode(data, Stage3Mode.TEST_D).x[-1] == 7.0


def test_handwriting_stage3_numpy_runtime_smoke():
    def spikes(rows, cols):
        return BinarySpikeSeries(Shape2D(rows, cols), bytes(rows * cols))

    def weights(dim0, dim1, dim2):
        return BinaryWeightWindows(Shape3D(dim0, dim1, dim2), array("f", [0.0] * (dim0 * dim1 * dim2)))

    def matrix(rows, cols):
        return MatrixF32(rows, cols, array("f", [0.0] * (rows * cols)))

    cfg = Stage3RuntimeConfig(
        n_ext=2,
        n_cm=4,
        n_bg=1,
        n_bg_total=6,
        n_sm_group=1,
        n_sm_total=2,
        n_mm=3,
        n_e=5,
        n_i=2,
        n_sig=2,
        n_pop=2,
        n_rcv=2,
        total_time_ms=20,
        dt_ms=0.1,
    )
    data = Stage3Data(
        sspk_ext=spikes(cfg.n_ext, 200),
        sspk_ext_e=spikes(cfg.n_e, 200),
        sspk_ext_i=spikes(cfg.n_i, 200),
        sspk_ext_mm=spikes(2 * cfg.n_mm, 200),
        sspk_sig=spikes(cfg.n_sig, 200),
        sspk_sm=spikes(cfg.n_sm_total, 200),
        ww_extcm1=weights(cfg.n_ext, 1, cfg.n_cm),
        ww_extcm2=weights(cfg.n_ext, 1, cfg.n_cm),
        ww_extcm3=weights(cfg.n_ext, 1, cfg.n_cm),
        a1_flat=[],
        a2_flat=[],
        a3_flat=[],
        x1y1=type("Dummy", (), {"x": [0.12], "y": [0.43]})(),
        x2y2=type("Dummy", (), {"x": [0.13], "y": [0.44]})(),
        x3y3=type("Dummy", (), {"x": [0.14], "y": [0.45]})(),
        c_emm1=matrix(cfg.n_mm, cfg.n_e),
        c_emm2=matrix(cfg.n_mm, cfg.n_e),
        c_mmbg1=matrix(cfg.n_bg_total, cfg.n_mm),
        c_mmbg2=matrix(cfg.n_bg_total, cfg.n_mm),
        w_mmbg1=matrix(cfg.n_bg_total, cfg.n_mm),
        w_mmbg2=matrix(cfg.n_bg_total, cfg.n_mm),
        c_cm1e=matrix(cfg.n_e, cfg.n_cm),
        c_cm2e=matrix(cfg.n_e, cfg.n_cm),
        c_cm3e=matrix(cfg.n_e, cfg.n_cm),
        c_sig_e=matrix(cfg.n_e, cfg.n_sig),
        w_ee=matrix(cfg.n_e, cfg.n_e),
    )
    result = run_stage3_numpy(cfg, data, Stage3Mode.TEST, max_steps=200)
    assert result.simulated_steps == 200
    assert len(result.pop_spk) == 2 * cfg.n_pop
    assert sum(result.pop_spk) == 0

    train_result = run_stage3_numpy(cfg, data, Stage3Mode.TRAIN_G, max_steps=200)
    assert train_result.simulated_steps == 200
    assert train_result.w_mmbg1.shape == (cfg.n_bg_total, cfg.n_mm)
    assert train_result.w_mmbg2.shape == (cfg.n_bg_total, cfg.n_mm)


def test_handwriting_stage3_framework_python_migration_smoke(tmp_path):
    input_dir = _external_data_dir("handwriting_stage3_framework")
    result = run_stage3_framework(input_dir, tmp_path / "stage3_framework", "test", max_steps=200)

    assert result.simulated_steps == 200
    assert len(result.pop_spk) == 16
    assert (tmp_path / "stage3_framework" / "QQ.tsv").exists()
    assert (tmp_path / "stage3_framework" / "pop_spk.tsv").exists()


def test_baseline_outer_dynamic_spike_counter_smoke():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2))
    network.add_outer_dynamic(nb.OuterDynamic.spike_counter(name="counter_sink", slot_count=3, type_count=2))
    network.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=[0, 1, 0],
            target_outer_dynamic=0,
            target_joint=[1, 1, 2],
            synapse_type=[0, 1, 0],
            weight=[2.0, 3.5, 4.0],
            delay=1,
        )
    )

    sim = nb.Simulation(network, nb.SimulationConfig(steps=8, timestep=1.0))
    sim.init()
    sim.add_external_spikes([0, 0], [0, 1])
    sim.run(4)

    snapshot = sim.outer_dynamic_spike_counter_snapshot("counter_sink")
    assert snapshot["slot_count"] == 3
    assert snapshot["spike_counts"][1] == 2
    assert snapshot["weighted_sums"][1] == pytest.approx(5.5)
    assert snapshot["spike_counts_by_type"][0][1] == 1
    assert snapshot["weighted_sums_by_type"][0][1] == pytest.approx(2.0)
    assert snapshot["spike_counts_by_type"][1][1] == 1
    assert snapshot["weighted_sums_by_type"][1][1] == pytest.approx(3.5)
    assert snapshot["spike_counts"][2] == 1
    assert snapshot["weighted_sums"][2] == pytest.approx(4.0)

    sim.clear_outer_dynamic_spike_counter_slot("counter_sink", 1)
    snapshot = sim.outer_dynamic_spike_counter_snapshot("counter_sink")
    assert snapshot["spike_counts"][1] == 0
    assert snapshot["weighted_sums"][1] == pytest.approx(0.0)
    assert snapshot["spike_counts"][2] == 1

    sim.clear_outer_dynamic_spike_counter("counter_sink")
    snapshot = sim.outer_dynamic_spike_counter_snapshot("counter_sink")
    assert snapshot["spike_counts"][2] == 0
    assert snapshot["weighted_sums"][2] == pytest.approx(0.0)


def test_cerebellum_framework_python_migration_smoke(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    cfg = CerebellumConfig(sample_count=4, nn=2, nv=2, n_pc=4, n_cf=4, n_dcn=4)
    summary = run_cerebellum_framework(cfg, tmp_path / "cerebellum")

    assert summary["samples"] == 2
    assert summary["outer_dynamics"] == 1
    assert summary["final_time_step"] == 400
    assert summary["final_q1"] == pytest.approx(0.9198566746536696)
    assert summary["final_q2"] == pytest.approx(2.316644660806827)
    assert (tmp_path / "cerebellum" / "joint_state.tsv").exists()
    assert (tmp_path / "cerebellum" / "summary.tsv").exists()


def test_zmqcommunication_protocol_helpers():
    header, payload = pack_spike_batch(12, [SpikeRecord(3, 2.4, 0.2), SpikeRecord(4, 2.6, 0.2)])
    time_step, spikes = unpack_spike_batch(header, payload)

    assert time_step == 12
    assert [spike.neuron for spike in spikes] == [3, 4]
    assert spikes[0].time == pytest.approx(2.4)
    assert spikes[1].base_timestep == pytest.approx(0.2)

    empty_header, empty_payload = pack_spike_batch(20, [])
    empty_time_step, empty_spikes = unpack_spike_batch(empty_header, empty_payload)
    assert empty_time_step == 20
    assert empty_spikes == []


def test_zmqcommunication_client_python_migration_smoke(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    summary = run_zmqcommunication_client(tmp_path / "zmq_client", steps=50)

    assert summary["steps"] == 50
    assert summary["enable_zmq"] == 0
    assert summary["buffered_output_spikes"] >= 1
    assert summary["monitor_spike_bytes"] > 0
    assert summary["monitor_state_bytes"] > 0


def test_baseline_dense_subnetwork_export_real():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_double(2, random_sigma=nb.float32_array4([1.0, 0.0, 0.0, 0.0])))
    network.add_layer(
        nb.NeuronLayer.lif_double(
            4,
            dense_name="export_dense_subnetwork",
            random_sigma=nb.float32_array4([2.0, 0.0, 0.0, 0.0]),
        )
    )
    network.add_layer(
        nb.NeuronLayer(
            "TimeDrivenLIF_Exponential_double",
            3,
            update_timestep=2,
            parameters={
                "dense_subnetwork_name": "export_dense_subnetwork",
                "random_sigma": nb.float32_array4([3.0, 0.0, 0.0, 0.0]),
            },
        )
    )
    network.add_layer(nb.NeuronLayer.lif_decay(2, output=True, random_sigma=nb.float32_array3([0.0, 0.0, 0.0])))
    network.connect(_all_to_all(0, 1, 1, 2, weight=20.0, delay=1))
    network.connect(_all_to_all(1, 2, 3, 4, weight=12.0, delay=1))
    network.connect(_all_to_all(3, 4, 7, 3, weight=8.0, delay=2))
    network.connect(_all_to_all(7, 3, 10, 2, weight=0.60, delay=3))

    sim = nb.Simulation(network, nb.SimulationConfig(steps=48, timestep=0.1))
    sim.init()
    assert sim.dense_subnetwork_count == 1
    assert sim.dense_subnetwork_name(0) == "export_dense_subnetwork"
    assert sim.find_dense_subnetwork("export_dense_subnetwork") == 0
    sim.set_dense_subnetwork_full_firing_export_enabled("export_dense_subnetwork")

    sim.add_external_spikes([0, 1, 2], [0, 0, 0])
    sim.run(30)
    snapshot = sim.dense_subnetwork_snapshot("export_dense_subnetwork")
    weights = sim.dense_subnetwork_weights(0)
    assert snapshot["name"] == "export_dense_subnetwork"
    assert snapshot["time_step"] > 0
    assert snapshot["cuda_build_enabled"]
    assert snapshot["gpu_backend_ready"]
    assert snapshot["carlsim_like_gpu_active"]
    assert len(snapshot["membrane_v"]) == 7
    assert len(snapshot["gexc"]) == len(snapshot["membrane_v"])
    assert len(snapshot["ginh"]) == len(snapshot["membrane_v"])
    assert len(snapshot["fired"]) == len(snapshot["membrane_v"])
    assert len(snapshot["synaptic_weights"]) == len(weights)
    assert len(snapshot["synaptic_learning_pending_dwt"]) == len(weights)
    assert any(value > 0.0 for value in snapshot["gexc"])

    sim.reset_dense_subnetwork("export_dense_subnetwork")
    reset_snapshot = sim.dense_subnetwork_snapshot(0)
    assert sum(reset_snapshot["fired"]) == 0
    assert reset_snapshot["full_firing_ids"] == []


def test_baseline_inputconv_poisson_dense_real(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    output_count = 512
    dense_name = "inputconv_poisson_dense_demo"
    network = nb.Network()
    network.add_layer(
        nb.NeuronLayer.poisson_rate(
            output_count,
            dense_name=dense_name,
            rate_bias_hz=0.0,
            rate_gain_hz_per_current=1.0,
            dense_steps_to_keep=128,
        )
    )
    network.add_input_conv(
        nb.InputConv.v1_bar(
            width=8,
            height=8,
            channels=1,
            speed=1.5,
            bar_width=2,
            motion_period_steps=16,
            output_target="dense_subnetwork",
            target_dense_subnetwork_name=dense_name,
            output_source_indices=list(range(output_count)),
            output_target_neuron_ids=list(range(output_count)),
            output_pending_channel=2,
        )
    )

    monitor_dir = tmp_path / "inputconv_poisson_dense"
    sim = nb.Simulation(network, nb.SimulationConfig(steps=40, timestep=1.0))
    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=monitor_dir,
            all_neurons=True,
            record_state=True,
            record_spikes=True,
            record_pending_channels=True,
            record_weights=False,
            monitor_all_inputconv=True,
            record_inputconv_outputs=True,
            record_inputconv_inputs=True,
        )
    )
    sim.run(32)
    sim.flush()

    assert sim.input_conv_count == 1
    assert sim.input_conv_output_count(0) == output_count
    assert len(sim.input_conv_output(0)) == output_count
    assert len(sim.input_conv_rate_maps(0, width=8, height=8, directions=8)) == 8
    snapshot = sim.dense_subnetwork_snapshot(dense_name)
    assert snapshot["name"] == dense_name
    assert len(snapshot["membrane_v"]) == output_count

    result = sim.result()
    assert result.path == monitor_dir
    assert result.file_sizes()["inputconv_outputs"] > 0
    assert result.file_sizes()["inputconv_inputs"] > 0
    assert result.file_sizes()["pending_channels"] > 0
    assert any(float(row["value"]) > 0.0 for row in result.iter_rows("inputconv_outputs", limit=output_count))


def test_input_conv_dynamic_frame_queue_native():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    for _ in range(2):
        network.add_input_conv(
            nb.InputConv.v1_bar(
                width=2,
                height=2,
                channels=1,
                input_frame_missing_policy="hold_last",
                input_frame_max_lag_steps=4,
            )
        )

    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))
    sim.add_input_conv_frames(
        "camera_bus",
        [
            nb.InputConvFrame(
                time_step=0,
                source_camera_index=0,
                width=2,
                height=2,
                channels=1,
                bytes=bytes([10, 10, 10, 10]),
            ),
            nb.InputConvFrame(
                time_step=0,
                source_camera_index=1,
                width=2,
                height=2,
                channels=1,
                bytes=bytes([20, 20, 20, 20]),
            ),
        ],
    )
    sim.bind_input_conv_frame_source(0, "camera_bus", source_camera_index=0)
    sim.bind_input_conv_frame_source(1, "camera_bus", source_camera_index=1)
    assert sim.has_input_conv_frame_source_binding(0)
    assert sim.has_input_conv_frame_source_binding(1)
    status = sim.input_conv_frame_source_status("camera_bus")
    assert status["received_frames"] == 2
    assert status["queued_frames"] == 2
    assert status["consumed_frames"] == 0
    assert status["failed_requests"] == 0

    sim.run(1)

    assert sim.input_conv_input(0) == [10.0, 10.0, 10.0, 10.0]
    assert sim.input_conv_input(1) == [20.0, 20.0, 20.0, 20.0]
    status = sim.input_conv_frame_source_status(0)
    assert status["consumed_frames"] == 2
    assert status["queued_frames"] == 0
    assert status["last_consumed_time_step"] == 0
    assert status["last_source_camera_index"] in (0, 1)

    sim.reset()
    sim.run(1)
    assert sim.input_conv_input(0) == [0.0, 0.0, 0.0, 0.0]


def test_input_conv_zmq_frame_source_native():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")
    pytest.importorskip("zmq")
    received_requests: list[nb.InputConvFrameRequest] = []

    def provide_frame(request: nb.InputConvFrameRequest) -> nb.InputConvFrame:
        received_requests.append(request)
        return nb.InputConvFrame(
            time_step=request.time_step,
            source_camera_index=0,
            width=2,
            height=2,
            channels=1,
            bytes=bytes([77, 77, 77, 77]),
        )

    network = nb.Network()
    network.add_input_conv(
        nb.InputConv.v1_bar(
            width=2,
            height=2,
            channels=1,
            input_frame_missing_policy="zero",
        )
    )
    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))
    with nb.InputConvFrameServer(provide_frame) as server:
        source_index = sim.add_input_conv_frame_server_source("sync_camera", server, bind_to=0)
        assert source_index == 0
        sim.run(1)
        source_status = sim.input_conv_frame_source_status("sync_camera")
        server_summary = server.summary()

    assert server_summary["received_requests"] >= 1
    assert server_summary["sent_frames"] >= 1
    assert source_status["requested_frames"] >= 1
    assert source_status["received_frames"] >= 1
    assert source_status["failed_requests"] == 0
    assert source_status["last_received_time_step"] >= 0
    assert source_status["last_error"] == ""
    assert received_requests
    assert received_requests[0].time_step == 0
    assert received_requests[0].width == 2
    assert received_requests[0].height == 2
    assert sim.input_conv_input(0) == [77.0, 77.0, 77.0, 77.0]


def test_input_conv_zmq_async_frame_source_native():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")
    pytest.importorskip("zmq")

    topic = "inputconv_bar"

    network = nb.Network()
    network.add_input_conv(
        nb.InputConv.v1_bar(
            width=2,
            height=2,
            channels=1,
            input_frame_missing_policy="hold_last",
            input_frame_max_lag_steps=4,
        )
    )
    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))
    frame = nb.InputConvFrame(
        time_step=0,
        source_camera_index=0,
        width=2,
        height=2,
        channels=1,
        bytes=bytes([88, 88, 88, 88]),
    )

    with nb.InputConvFramePublisher(topic=topic) as publisher:
        source_index = sim.add_zmq_async_input_conv_frame_source(
            "async_camera",
            subscribe_address="127.0.0.1",
            subscribe_port=publisher.port,
            topic=publisher.topic,
            max_buffered_frames_per_camera=64,
        )
        assert source_index == 0
        sim.bind_input_conv_frame_source(0, "async_camera")
        publisher.publish_frame_until_received(
            frame,
            sim,
            "async_camera",
            repeat=1,
            interval_s=0.0,
            timeout_s=2.0,
            poll_interval_s=0.01,
            max_attempts=80,
        )
        pre_run_status = sim.input_conv_frame_source_status("async_camera")
        sim.run(1)
        source_status = sim.input_conv_frame_source_status("async_camera")
        publisher_summary = publisher.summary()

    assert publisher_summary["published_frames"] >= 1
    assert publisher_summary["published_messages"] >= 1
    assert pre_run_status["running"]
    assert pre_run_status["received_frames"] >= 1
    assert source_status["received_frames"] >= 1
    assert source_status["consumed_frames"] >= 1
    assert source_status["failed_requests"] == 0
    assert source_status["last_consumed_time_step"] == 0
    assert sim.input_conv_input(0) == [88.0, 88.0, 88.0, 88.0]


def test_input_conv_dynamic_frame_negative_native():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    network.add_input_conv(
        nb.InputConv.v1_bar(
            width=2,
            height=2,
            channels=1,
            input_frame_missing_policy="zero",
            input_frame_max_lag_steps=1,
        )
    )
    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))
    sim.add_input_conv_frames(
        "camera_bus",
        [
            nb.InputConvFrame(
                time_step=0,
                source_camera_index=0,
                width=3,
                height=2,
                channels=1,
                bytes=bytes([55] * 6),
            )
        ],
    )
    sim.bind_input_conv_frame_source(0, "camera_bus")
    sim.run(1)
    assert sim.input_conv_input(0) == [0.0, 0.0, 0.0, 0.0]

    sim.reset()
    sim.run(2)
    sim.add_input_conv_frames(
        "camera_bus",
        [
            nb.InputConvFrame(
                time_step=0,
                source_camera_index=0,
                width=2,
                height=2,
                channels=1,
                bytes=bytes([66] * 4),
            )
        ],
    )
    sim.bind_input_conv_frame_source(0, "camera_bus")
    sim.run(1)
    assert sim.input_conv_input(0) == [0.0, 0.0, 0.0, 0.0]


def test_input_conv_zmq_bad_response_magic_native():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")
    zmq = pytest.importorskip("zmq")
    import threading

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.linger = 0
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    stop_server = threading.Event()

    def serve_bad_frame() -> None:
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not stop_server.is_set():
            events = dict(poller.poll(100))
            if socket not in events:
                continue
            socket.recv()
            frame = nb.InputConvFrame(
                time_step=0,
                source_camera_index=0,
                width=2,
                height=2,
                channels=1,
                bytes=bytes([99] * 4),
            )
            response = bytearray(nb.pack_input_conv_frame(frame))
            response[:4] = (0).to_bytes(4, "little")
            header_size = len(response) - len(frame.bytes)
            socket.send(bytes(response[:header_size]), flags=zmq.SNDMORE)
            socket.send(bytes(response[header_size:]))

    server = threading.Thread(target=serve_bad_frame, daemon=True)
    server.start()

    network = nb.Network()
    network.add_input_conv(
        nb.InputConv.v1_bar(width=2, height=2, channels=1, input_frame_missing_policy="zero")
    )
    sim = nb.Simulation(network, nb.SimulationConfig(steps=2, timestep=1.0))
    sim.add_zmq_input_conv_frame_source("bad_sync_camera", address="127.0.0.1", port=port)
    sim.bind_input_conv_frame_source(0, "bad_sync_camera")
    try:
        sim.run(1)
    finally:
        stop_server.set()
    server.join(timeout=3.0)
    socket.close(0)
    context.term()

    assert not server.is_alive()
    source_status = sim.input_conv_frame_source_status("bad_sync_camera")
    assert source_status["requested_frames"] >= 1
    assert source_status["received_frames"] == 0
    assert source_status["failed_requests"] >= 1
    assert "magic" in source_status["last_error"].lower()
    assert sim.input_conv_input(0) == [0.0, 0.0, 0.0, 0.0]


def test_input_conv_zmq_async_wrong_topic_recovers_native():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")
    pytest.importorskip("zmq")

    good_topic = "inputconv_good"
    bad_topic = "inputconv_bad"

    network = nb.Network()
    network.add_input_conv(
        nb.InputConv.v1_bar(
            width=2,
            height=2,
            channels=1,
            input_frame_missing_policy="zero",
            input_frame_max_lag_steps=4,
        )
    )
    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))
    bad_frame = nb.InputConvFrame(
        time_step=0,
        source_camera_index=0,
        width=2,
        height=2,
        channels=1,
        bytes=bytes([11] * 4),
    )
    good_frame = nb.InputConvFrame(
        time_step=0,
        source_camera_index=0,
        width=2,
        height=2,
        channels=1,
        bytes=bytes([44] * 4),
    )

    with nb.InputConvFramePublisher(topic=good_topic) as publisher:
        sim.add_zmq_async_input_conv_frame_source("async_camera", subscribe_address="127.0.0.1", subscribe_port=publisher.port, topic=good_topic)
        sim.bind_input_conv_frame_source(0, "async_camera")
        publisher.wait_ready(delay_s=0.2)
        publisher.publish_frame(bad_frame, repeat=20, interval_s=0.003, topic=bad_topic)
        publisher.wait_ready(delay_s=0.05)
        sim.run(1)
        wrong_topic_status = sim.input_conv_frame_source_status("async_camera")
        assert sim.input_conv_input(0) == [0.0, 0.0, 0.0, 0.0]

        publisher.publish_frame(good_frame, repeat=40, interval_s=0.003)
        publisher.wait_ready(delay_s=0.1)
        sim.run(1)
        recovered_status = sim.input_conv_frame_source_status("async_camera")

    assert sim.input_conv_input(0) == [44.0, 44.0, 44.0, 44.0]
    assert wrong_topic_status["received_frames"] == 0
    assert wrong_topic_status["queued_frames"] == 0
    assert recovered_status["received_frames"] >= 1
    assert recovered_status["consumed_frames"] >= 1


def test_baseline_dense_mixed(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    monitor_dir = tmp_path / "dense_mixed" / "with_relay"
    sim = nb.Simulation(build_dense_mixed_network(connect_relay_to_dense=True), nb.SimulationConfig(steps=300, timestep=0.1))
    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=monitor_dir,
            sample_interval_steps=1,
            flush_interval_steps=10,
            all_neurons=True,
            record_state=True,
            record_spikes=True,
            record_pending_channels=True,
            record_weights=False,
            record_outer_dynamic_state=False,
            record_inputconv_outputs=False,
        )
    )
    sim.add_external_spikes([0, 10, 40], [0, 0, 0])
    for _ in range(300):
        sim.run(1)
    sim.flush()

    snapshot = sim.dense_subnetwork_snapshot("mixed_dense_subnetwork")
    assert sim.dense_subnetwork_count == 1
    assert snapshot["name"] == "mixed_dense_subnetwork"
    assert len(snapshot["membrane_v"]) == 7
    assert len(snapshot["synaptic_weights"]) == 12
    assert snapshot["gpu_backend_ready"]
    assert snapshot["carlsim_like_gpu_active"]
    result = sim.result()
    assert result.file_sizes()["neuron_state"] > 0
    assert result.file_sizes()["pending_channels"] > 0


def test_baseline_dense_mixed_current():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    sim = nb.Simulation(build_dense_mixed_current_network(), nb.SimulationConfig(steps=64, timestep=1.0))
    sim.init()
    sim.add_external_spikes([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0])
    sim.add_external_currents([0], [1], [22.0])
    sim.run(40)

    output_spikes = sim.output_spikes()
    snapshot = sim.dense_subnetwork_snapshot("mixed_current_dense_demo")
    assert sim.dense_subnetwork_count == 1
    assert snapshot["name"] == "mixed_current_dense_demo"
    assert len(snapshot["membrane_v"]) == 7
    assert len(snapshot["synaptic_weights"]) == 12
    assert snapshot["gpu_backend_ready"]
    assert snapshot["carlsim_like_gpu_active"]
    assert any(value > 0.0 for value in snapshot["gexc"])
    # Dense interface spikes are committed on the next Dense update after the
    # producing simulation step has completed.  This preserves deterministic
    # behavior when the producer and Dense subnetwork use different queues.
    assert output_spikes == [{"time": 5, "neuron_id": 2}, {"time": 5, "neuron_id": 3}]
    assert sim.output_spikes() == []


def test_baseline_dense_run_no_debug():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    sim = nb.Simulation(build_dense_run_no_debug_network(), nb.SimulationConfig(steps=1200, timestep=1.0))
    sim.init()
    sim.add_external_spikes([0, 20, 40, 60, 80, 100], [0, 0, 0, 0, 0, 0])
    sim.run(1000)

    output_spikes = sim.output_spikes()
    readout0 = sim.neuron_state(8)
    readout1 = sim.neuron_state(9)
    assert len(output_spikes) == 4
    assert readout0["main_neuron_id"] >= 0
    assert readout1["main_neuron_id"] >= 0
    assert readout0["state_variables"][1] == pytest.approx(2.945720370614717e-32)
    assert readout1["state_variables"][1] == pytest.approx(2.945720370614717e-32)


def test_baseline_dense_public_smoke_checks():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1, dense_subnetwork_name="bad_dense"))
    with pytest.raises(RuntimeError, match="cannot be placed inside dense subnetwork"):
        nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))

    network = nb.Network()
    network.add_layer(
        nb.NeuronLayer(
            "TimeDrivenLIF_Exponential_double",
            1,
            output=True,
            parameters={"dense_subnetwork_name": "bad_output_dense"},
        )
    )
    with pytest.raises(RuntimeError, match="cannot be marked isOutput=true"):
        nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))

    network = nb.Network()
    network.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 1, parameters={"dense_subnetwork_name": "dense_a"}))
    network.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 1, parameters={"dense_subnetwork_name": "dense_b"}))
    network.connect(nb.Connection(0, 1, weight=7.0, max_weight=7.0, delay=2))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))
    sim.init()
    assert sim.dense_subnetwork_count == 2
    assert sim.dense_subnetwork_name(0) == "dense_a"
    assert sim.dense_subnetwork_name(1) == "dense_b"


def test_baseline_ei_main_small_run_and_batch_state():
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    connectivity_dir = _external_data_dir("ei_connectivity")
    if not connectivity_dir.exists():
        pytest.skip("EI connectivity reference data is not available")

    sim = nb.Simulation(build_main_ei_program(load_connectivity(connectivity_dir)), nb.SimulationConfig(steps=10, timestep=0.1))
    sim.init()
    sim.add_external_currents([0], [NEURON_COUNT], [12.0])
    sim.run(10)
    states = sim.neuron_states([0, 1, NEURON_COUNT - 1])
    assert [state["original_neuron_id"] for state in states] == [0, 1, NEURON_COUNT - 1]
    assert all(len(state["state_variables"]) >= 4 for state in states)


def test_baseline_pattern_motion_quick_inputconv_dense(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    spec = PatternMotionSpec(
        width=4,
        height=4,
        frames_per_direction=1,
        v1_update_interval_steps=2,
        pds_grid_width=2,
        pds_grid_height=2,
        lip_grid_width=2,
        lip_grid_height=1,
    )
    stimulus_path = write_stimulus_file(tmp_path / "selected_stimulus.dat", spec, mode="grating", selected_block=0)
    sim = nb.Simulation(build_pattern_motion_network(spec, stimulus_path), nb.SimulationConfig(steps=2, timestep=1.0))
    sim.init()
    sim.set_dense_subnetwork_full_firing_export_enabled(spec.dense_name)
    sim.run(2)

    snapshot = sim.dense_subnetwork_snapshot(spec.dense_name)
    assert sim.dense_subnetwork_count == 1
    assert sim.input_conv_count == 1
    assert sim.input_conv_output_count(0) == spec.v1_count
    assert len(sim.input_conv_output(0)) == spec.v1_count
    assert len(sim.input_conv_input(0)) == spec.width * spec.height
    assert len(sim.input_conv_rate_maps(0)) == 8
    assert snapshot["name"] == spec.dense_name
    assert len(snapshot["membrane_v"]) == spec.total_dense_neurons
    assert len(snapshot["local_to_original_neuron_ids"]) == spec.total_dense_neurons
    assert snapshot["gpu_backend_ready"]


def test_pattern_motion_dynamic_input_sources(tmp_path):
    if not nb.backend_info()["native_extension_loaded"]:
        pytest.skip("native extension is not built")

    queue_summary = run_pattern_motion(
        "grating",
        0,
        tmp_path / "pattern_motion_queue",
        steps=2,
        input_source="queue",
    )
    assert queue_summary["input_source"] == "queue"
    assert queue_summary["dense_subnetwork_count"] == 1
    assert 0 <= int(queue_summary["v1_rate_winner"]) < 8
    assert (tmp_path / "pattern_motion_queue" / "summary.txt").exists()

    pytest.importorskip("zmq")
    zmq_summary = run_pattern_motion(
        "grating",
        0,
        tmp_path / "pattern_motion_zmq_async",
        steps=2,
        input_source="zmq-async",
        frame_topic="test_pattern_motion_inputconv",
    )
    assert zmq_summary["input_source"] == "zmq-async"
    assert zmq_summary["dense_subnetwork_count"] == 1
    assert zmq_summary["v1_rate_winner"] == queue_summary["v1_rate_winner"]
    assert zmq_summary["v1_current_update_mean_winner"] == queue_summary["v1_current_update_mean_winner"]
    assert (tmp_path / "pattern_motion_zmq_async" / "summary.txt").exists()
