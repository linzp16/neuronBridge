import json
import neuronbridge as nb

def provide(req):
    return nb.InputConvFrame(req.time_step, req.source_camera_index, 2, 2, 1, bytes([1, 2, 3, 4]), req.preferred_pixel_format)

rows = []
for i in range(10):
    frame = nb.InputConvFrame(i, i % 2, 2, 2, 1, bytes([1, 2, 3, 4]))
    with nb.InputConvFramePublisher(topic=f"lifecycle_{i}") as publisher:
        publisher.publish_frame(frame, repeat=2)
        rows.append({"publisher_closed_inside": publisher.closed, "published": publisher.published_messages})
    rows[-1]["publisher_closed_after"] = publisher.closed
    with nb.InputConvFrameServer(provide) as server:
        with nb.InputConvFrameClient(port=server.port) as client:
            response = client.request_frame(nb.InputConvFrameRequest(i, 0, i % 2, 2, 2, 1))
            rows[-1].update({"client_closed_inside": client.closed, "server_running_inside": server.running, "response_step": response.time_step})
        rows[-1]["client_closed_after"] = client.closed
    rows[-1]["server_closed_after"] = server.closed

assert all(row["publisher_closed_after"] and row["client_closed_after"] and row["server_closed_after"] for row in rows)
print(json.dumps({"ok": True, "iterations": len(rows), "rows": rows}, indent=2))
