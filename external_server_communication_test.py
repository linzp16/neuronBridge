from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import neuronbridge as nb

OUT = Path(r"D:\testfile\communication_external_server")
OUT.mkdir(parents=True, exist_ok=True)


def make_frame(step: int, source_camera_index: int = 0) -> nb.InputConvFrame:
    return nb.InputConvFrame(
        time_step=step,
        source_camera_index=source_camera_index,
        width=2,
        height=2,
        channels=1,
        pixel_format="uint8_gray",
        bytes=bytes([(step + i) % 256 for i in range(4)]),
    )


def make_inputconv_network() -> nb.Network:
    net = nb.Network()
    net.add_input_conv(nb.InputConv.v1_grating(width=2, height=2, channels=1, output_target="main_network"))
    return net


def sync_python_req_rep() -> dict:
    def provider(request: nb.InputConvFrameRequest) -> nb.InputConvFrame:
        return make_frame(request.time_step, request.source_camera_index)

    with nb.InputConvFrameServer(provider, poll_timeout_ms=20) as server:
        with nb.InputConvFrameClient(port=server.port, receive_timeout_ms=2000) as client:
            rows = []
            for step in range(100):
                response = client.request_frame(nb.InputConvFrameRequest(step, step % 2, step % 3, 2, 2, 1))
                expected = bytes([(step + i) % 256 for i in range(4)])
                assert response.time_step == step
                assert response.source_camera_index == step % 3
                assert response.bytes == expected
                rows.append(response.time_step)
            result = {
                "requests": len(rows),
                "server": server.summary(),
                "client": client.summary(),
                "all_payloads_valid": len(rows) == 100,
                "closed_after_context": False,
            }
        result["client_closed_after_context"] = client.closed
    result["server_closed_after_context"] = server.closed
    return result


def sync_native_external_server() -> dict:
    def provider(request: nb.InputConvFrameRequest) -> nb.InputConvFrame:
        return make_frame(request.time_step, request.source_camera_index)

    with nb.InputConvFrameServer(provider, poll_timeout_ms=10) as server:
        sim = nb.Simulation(make_inputconv_network(), nb.SimulationConfig(steps=40, timestep=1.0)).init()
        sim.add_input_conv_frame_server_source("sync_external", server, bind_to=0)
        sim.run(40)
        status = sim.input_conv_frame_source_status("sync_external")
        result = {"source_status": status, "server": server.summary()}
    result["server_closed"] = server.closed
    result["received_requests_positive"] = int(result["server"]["received_requests"]) > 0
    assert result["received_requests_positive"], result
    return result


def async_native_pub_sub() -> dict:
    with nb.InputConvFramePublisher(topic="external_async_test") as publisher:
        sim = nb.Simulation(make_inputconv_network(), nb.SimulationConfig(steps=40, timestep=1.0)).init()
        sim.add_zmq_async_input_conv_frame_source(
            "async_external",
            subscribe_address="127.0.0.1",
            subscribe_port=publisher.port,
            topic=publisher.topic,
            max_buffered_frames_per_camera=64,
        )
        sim.bind_input_conv_frame_source(0, "async_external")
        published = []
        for step in range(20):
            frame = make_frame(step)
            publisher.publish_frame_until_received(
                frame,
                sim,
                "async_external",
                repeat=1,
                timeout_s=2.0,
                poll_interval_s=0.01,
                max_attempts=20,
            )
            published.append(step)
        status = sim.input_conv_frame_source_status("async_external")
        sim.run(40)
        status = sim.input_conv_frame_source_status("async_external")
        result = {"published_frames": len(published), "publisher": publisher.summary(), "source_status": status}
    result["publisher_closed"] = publisher.closed
    result["received_frames_positive"] = int(result["source_status"]["received_frames"]) > 0
    result["received_not_excessive"] = int(result["source_status"]["received_frames"]) <= len(published)
    result["no_dropped_frames"] = int(result["source_status"].get("dropped_frames", 0)) == 0
    assert result["received_frames_positive"] and result["received_not_excessive"] and result["no_dropped_frames"], result
    return result


def main() -> int:
    rows = []
    for name, fn in (("sync_python_req_rep", sync_python_req_rep), ("sync_native_external_server", sync_native_external_server), ("async_native_pub_sub", async_native_pub_sub)):
        try:
            rows.append({"name": name, "status": "PASS", "detail": fn()})
        except Exception as exc:
            rows.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    result = {"ok": all(row["status"] == "PASS" for row in rows), "cases": rows}
    (OUT / "external_server_communication_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


raise SystemExit(main())
