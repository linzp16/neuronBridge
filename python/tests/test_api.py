import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import neuronbridge as nb
from neuronbridge.frames import expected_payload_bytes, input_conv_pixel_format


def test_reset_has_no_weight_policy_switch():
    signature = inspect.signature(nb.Simulation.reset)
    assert list(signature.parameters) == ["self"]


def test_remaining_typed_parameter_helpers():
    values = (
        (nb.float64(1.25), "float64", 1.25),
        (nb.int32_list([1, 2, 3]), "int32_list", [1, 2, 3]),
        (nb.float32_list([1, 2.5]), "float32_list", [1.0, 2.5]),
        (nb.float32_array5([1, 2, 3, 4, 5]), "float32_array5", [1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    for parameter, kind, expected in values:
        assert parameter.kind == kind
        assert parameter.value == expected
        assert parameter.__neuronbridge_param_kind__ == kind


def test_input_conv_pixel_format_and_payload_helpers():
    assert input_conv_pixel_format("gray") is nb.InputConvPixelFormat.UINT8_GRAY
    assert input_conv_pixel_format("RGB") is nb.InputConvPixelFormat.UINT8_RGB
    assert input_conv_pixel_format(5) is nb.InputConvPixelFormat.FLOAT32_CHW
    assert nb.InputConvPixelFormat.FLOAT32_CHW.native_name == "float32_chw"
    assert expected_payload_bytes(2, 3, 1, "uint8_gray") == 6
    assert expected_payload_bytes(2, 3, 1, "uint8_rgb") == 18
    assert expected_payload_bytes(2, 3, 2, "float32_hwc") == 48
    with pytest.raises(ValueError, match="unsupported InputConv pixel format"):
        input_conv_pixel_format("not-a-format")


def test_debug_monitor_result_metadata_paths_and_pandas_delegation(tmp_path: Path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    table = tmp_path / "spikes.csv"
    table.write_text("time_step,global_neuron_id\n1,7\n", encoding="utf-8")
    result = nb.open_debug_monitor(tmp_path)

    assert result.meta_path == tmp_path / "meta.json"
    assert result.meta() == {"version": 1}
    assert result.table_path("spikes") == table
    with pytest.raises(FileNotFoundError):
        result.table_path("weights")

    calls = []
    fake_pandas = SimpleNamespace(
        read_csv=lambda path, **kwargs: calls.append((path, kwargs)) or "frame"
    )
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    assert result.to_pandas("spikes", nrows=1) == "frame"
    assert calls == [(table, {"nrows": 1})]


def test_public_metadata_and_model_description_objects():
    info = nb.backend_info()
    assert info["package"] == "neuronbridge"
    assert info["binding"] == "pybind11"
    assert info["api_stage"] == "python-api"

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2))
    network.add_layer(nb.NeuronLayer.lif_double(1, V_th=nb.float32(-50.0)))
    network.connect(nb.Connection(source=[0, 1], target=2, weight=[1.0, 2.0]))
    assert network.neuron_count == 3
    assert network.to_dict()["connections"][0]["target"] == [2, 2]


@pytest.mark.parametrize(
    "factory",
    [
        nb.NeuronLayer.input_spike,
        nb.NeuronLayer.input_current,
        nb.NeuronLayer.lif_decay,
        nb.NeuronLayer.lif_double,
        nb.NeuronLayer.poisson_rate,
    ],
)
def test_neuron_layer_helpers_forward_structural_metadata(factory):
    layer = factory(
        2,
        update_timestep=3,
        monitored=True,
        output=True,
        communication_input=True,
    )
    assert layer.update_timestep == 3
    assert layer.monitored is True
    assert layer.output is True
    assert layer.communication_input is True
    assert "update_timestep" not in layer.parameters
    assert "monitored" not in layer.parameters
    assert "output" not in layer.parameters
    assert "communication_input" not in layer.parameters


def test_input_conv_sync_and_async_protocol_round_trip():
    request = nb.InputConvFrameRequest(
        time_step=12,
        inputconv_index=1,
        source_camera_index=3,
        width=2,
        height=2,
        channels=1,
        preferred_pixel_format="uint8_gray",
        max_lag_steps=4,
    )
    decoded_request = nb.unpack_input_conv_frame_request(
        nb.pack_input_conv_frame_request(request)
    )
    assert decoded_request == request

    frame = nb.InputConvFrame(
        time_step=12,
        source_camera_index=3,
        width=2,
        height=2,
        channels=1,
        pixel_format="uint8_gray",
        bytes=b"\x01\x02\x03\x04",
    )
    for asynchronous in (False, True):
        decoded = nb.unpack_input_conv_frame(
            nb.pack_input_conv_frame(frame, async_message=asynchronous),
            async_message=asynchronous,
        )
        assert decoded == frame


def test_input_conv_protocol_rejects_invalid_and_truncated_payloads():
    with pytest.raises(ValueError, match="payload"):
        nb.InputConvFrame(0, 0, 2, 2, 1, b"\x00", "uint8_gray")
    with pytest.raises(ValueError, match="magic"):
        nb.unpack_input_conv_frame_request(b"\x00" * 36)

    frame = nb.InputConvFrame(0, 0, 2, 2, 1, b"\x00" * 4)
    payload = nb.pack_input_conv_frame(frame, async_message=True)
    with pytest.raises(ValueError, match="incomplete"):
        nb.unpack_input_conv_frame(payload[:-1], async_message=True)


def test_weight_and_outer_dynamic_visualization_api(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    (tmp_path / "weights.csv").write_text(
        "time_step,component_kind,component_index,component_name,synapse_id,value\n"
        "0,MainNetwork,0,main_network,0,1.0\n"
        "1,MainNetwork,0,main_network,0,2.0\n",
        encoding="utf-8",
    )
    (tmp_path / "outer_dynamic_state.csv").write_text(
        "time_step,component_index,component_name,field_name,index,value\n"
        "0,0,arm,q,0,1.0\n1,0,arm,q,0,2.0\n"
        "0,0,arm,qv,0,0.1\n1,0,arm,qv,0,0.2\n",
        encoding="utf-8",
    )
    result = nb.open_debug_monitor(tmp_path)
    weight_ax = result.plot_weight_trace("main_network", [0])
    phase_ax = result.plot_outer_dynamic_phase_plane(outer_dynamic="arm")
    assert list(weight_ax.lines[0].get_ydata()) == [1.0, 2.0]
    assert list(phase_ax.lines[0].get_xdata()) == [1.0, 2.0]
    assert list(phase_ax.lines[0].get_ydata()) == [0.1, 0.2]
    plt.close("all")
