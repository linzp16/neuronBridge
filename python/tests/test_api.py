from pathlib import Path

import pytest

import neuronbridge as nb


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
