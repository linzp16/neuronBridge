"""ZMQ helpers for dynamic InputConv frame sources."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import threading
import time
from typing import Any

from .frames import (
    InputConvFrame,
    InputConvFrameRequest,
    expected_payload_bytes,
    pack_input_conv_frame,
    pack_input_conv_frame_request,
    unpack_input_conv_frame,
    unpack_input_conv_frame_request,
)


def _import_zmq() -> Any:
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError("InputConv ZMQ helpers require the 'communication' extra: pip install neuronbridge[communication]") from exc
    return zmq


@dataclass(slots=True)
class InputConvFramePublisherStats:
    published_frames: int = 0
    published_messages: int = 0
    last_time_step: int | None = None
    last_source_camera_index: int | None = None


class InputConvFramePublisher:
    """Publish async InputConv frames over the C++ PUB/SUB wire protocol."""

    def __init__(
        self,
        *,
        bind_address: str = "127.0.0.1",
        port: int = 0,
        topic: str = "inputconv_frame",
        linger_ms: int = 0,
        autostart: bool = True,
    ) -> None:
        self.bind_address = bind_address
        self.port = int(port)
        self.topic = str(topic)
        self.linger_ms = int(linger_ms)
        self._context: Any | None = None
        self._socket: Any | None = None
        self._closed = False
        self._stats = InputConvFramePublisherStats()
        if autostart:
            self.start()

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.bind_address}:{self.port}"

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def published_frames(self) -> int:
        return self._stats.published_frames

    @property
    def published_messages(self) -> int:
        return self._stats.published_messages

    @property
    def last_time_step(self) -> int | None:
        return self._stats.last_time_step

    @property
    def last_source_camera_index(self) -> int | None:
        return self._stats.last_source_camera_index

    def __enter__(self) -> "InputConvFramePublisher":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def start(self) -> "InputConvFramePublisher":
        if self._socket is not None:
            return self
        zmq = _import_zmq()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.linger = self.linger_ms
        if self.port <= 0:
            self.port = int(self._socket.bind_to_random_port(f"tcp://{self.bind_address}"))
        else:
            self._socket.bind(self.endpoint)
        self._closed = False
        return self

    def wait_ready(self, delay_s: float = 0.25) -> "InputConvFramePublisher":
        time.sleep(max(0.0, float(delay_s)))
        return self

    def wait_for_native_receiver(
        self,
        simulation: Any,
        source_name_or_index: str | int,
        *,
        previous_received_frames: int | None = None,
        min_received_frames: int | None = None,
        timeout_s: float = 2.0,
        poll_interval_s: float = 0.01,
    ) -> dict[str, Any]:
        """Wait until a native async frame source reports received frames."""

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        baseline = previous_received_frames
        if baseline is None and min_received_frames is None:
            baseline = int(simulation.input_conv_frame_source_status(source_name_or_index)["received_frames"])
        target = int(min_received_frames) if min_received_frames is not None else int(baseline or 0) + 1
        last_status: dict[str, Any] | None = None
        while True:
            last_status = dict(simulation.input_conv_frame_source_status(source_name_or_index))
            if int(last_status.get("received_frames", 0)) >= target:
                return last_status
            if time.monotonic() >= deadline:
                break
            time.sleep(max(0.001, float(poll_interval_s)))
        raise TimeoutError(
            f"InputConv frame source {source_name_or_index!r} did not reach received_frames={target}; "
            f"last status: {last_status}"
        )

    def publish_frame_until_received(
        self,
        frame: InputConvFrame,
        simulation: Any,
        source_name_or_index: str | int,
        *,
        repeat: int = 1,
        interval_s: float = 0.0,
        topic: str | None = None,
        timeout_s: float = 2.0,
        poll_interval_s: float = 0.01,
        max_attempts: int = 20,
    ) -> dict[str, Any]:
        """Publish a frame until the native async source reports it was received."""

        start_status = dict(simulation.input_conv_frame_source_status(source_name_or_index))
        baseline = int(start_status.get("received_frames", 0))
        attempts = max(1, int(max_attempts))
        per_attempt_timeout = max(0.001, float(timeout_s) / attempts)
        last_error: TimeoutError | None = None
        for _ in range(attempts):
            self.publish_frame(frame, repeat=repeat, interval_s=interval_s, topic=topic)
            try:
                return self.wait_for_native_receiver(
                    simulation,
                    source_name_or_index,
                    previous_received_frames=baseline,
                    timeout_s=per_attempt_timeout,
                    poll_interval_s=poll_interval_s,
                )
            except TimeoutError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise TimeoutError(f"InputConv frame source {source_name_or_index!r} did not receive a published frame")

    def publish_frame(
        self,
        frame: InputConvFrame,
        *,
        repeat: int = 1,
        interval_s: float = 0.0,
        topic: str | None = None,
    ) -> "InputConvFramePublisher":
        self.start()
        assert self._socket is not None
        zmq = _import_zmq()
        frame_topic = self.topic if topic is None else str(topic)
        payload = pack_input_conv_frame(frame, async_message=True)
        header_size = len(payload) - len(frame.bytes)
        repeats = max(1, int(repeat))
        for repeat_index in range(repeats):
            self._socket.send_string(frame_topic, flags=zmq.SNDMORE)
            self._socket.send(payload[:header_size], flags=zmq.SNDMORE)
            self._socket.send(payload[header_size:])
            self._stats.published_messages += 1
            if interval_s > 0.0 and repeat_index + 1 < repeats:
                time.sleep(float(interval_s))
        self._stats.published_frames += 1
        self._stats.last_time_step = frame.time_step
        self._stats.last_source_camera_index = frame.source_camera_index
        return self

    def publish_frames(
        self,
        frames: Iterable[InputConvFrame],
        *,
        repeat: int = 1,
        interval_s: float = 0.0,
        frame_interval_s: float = 0.0,
        topic: str | None = None,
    ) -> "InputConvFramePublisher":
        for frame in frames:
            self.publish_frame(frame, repeat=repeat, interval_s=interval_s, topic=topic)
            if frame_interval_s > 0.0:
                time.sleep(float(frame_interval_s))
        return self

    def summary(self) -> dict[str, int | str | bool | None]:
        return {
            "endpoint": self.endpoint,
            "bind_address": self.bind_address,
            "port": self.port,
            "topic": self.topic,
            "closed": self.closed,
            "published_frames": self.published_frames,
            "published_messages": self.published_messages,
            "last_time_step": self.last_time_step,
            "last_source_camera_index": self.last_source_camera_index,
        }

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._closed = True


FrameProvider = Callable[[InputConvFrameRequest], InputConvFrame | None]


@dataclass(slots=True)
class InputConvFrameServerStats:
    received_requests: int = 0
    sent_frames: int = 0
    provider_errors: int = 0
    last_request_time_step: int | None = None
    last_inputconv_index: int | None = None
    last_error: str = ""


class InputConvFrameServer:
    """Serve synchronous InputConv frames over the C++ REQ/REP wire protocol."""

    def __init__(
        self,
        frame_provider: FrameProvider | None = None,
        *,
        bind_address: str = "127.0.0.1",
        port: int = 0,
        linger_ms: int = 0,
        poll_timeout_ms: int = 100,
        autostart: bool = True,
    ) -> None:
        self.frame_provider = frame_provider
        self.bind_address = bind_address
        self.port = int(port)
        self.linger_ms = int(linger_ms)
        self.poll_timeout_ms = int(poll_timeout_ms)
        self._context: Any | None = None
        self._socket: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._closed = False
        self._stats = InputConvFrameServerStats()
        if autostart:
            self.start()

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.bind_address}:{self.port}"

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def received_requests(self) -> int:
        return self._stats.received_requests

    @property
    def sent_frames(self) -> int:
        return self._stats.sent_frames

    def __enter__(self) -> "InputConvFrameServer":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def start(self) -> "InputConvFrameServer":
        if self._thread is not None:
            return self
        zmq = _import_zmq()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.linger = self.linger_ms
        if self.port <= 0:
            self.port = int(self._socket.bind_to_random_port(f"tcp://{self.bind_address}"))
        else:
            self._socket.bind(self.endpoint)
        self._stop.clear()
        self._ready.set()
        self._closed = False
        self._thread = threading.Thread(target=self._serve_loop, name="InputConvFrameServer", daemon=True)
        self._thread.start()
        return self

    def wait_ready(self, timeout_s: float = 2.0) -> bool:
        return self._ready.wait(timeout=float(timeout_s))

    def _serve_loop(self) -> None:
        zmq = _import_zmq()
        assert self._socket is not None
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(self.poll_timeout_ms))
            if self._socket not in events:
                continue
            request_payload = self._socket.recv()
            try:
                request = unpack_input_conv_frame_request(request_payload)
                self._stats.received_requests += 1
                self._stats.last_request_time_step = request.time_step
                self._stats.last_inputconv_index = request.inputconv_index
                frame = self._provide_frame(request)
                response = pack_input_conv_frame(frame)
                header_size = len(response) - len(frame.bytes)
                self._socket.send(response[:header_size], flags=zmq.SNDMORE)
                self._socket.send(response[header_size:])
                self._stats.sent_frames += 1
                self._stats.last_error = ""
            except Exception as exc:  # Keep REQ/REP state valid by always replying.
                self._stats.provider_errors += 1
                self._stats.last_error = str(exc)
                fallback = self._zero_frame_from_request_payload(request_payload)
                response = pack_input_conv_frame(fallback)
                header_size = len(response) - len(fallback.bytes)
                self._socket.send(response[:header_size], flags=zmq.SNDMORE)
                self._socket.send(response[header_size:])

    def _provide_frame(self, request: InputConvFrameRequest) -> InputConvFrame:
        if self.frame_provider is None:
            return self._zero_frame(request)
        frame = self.frame_provider(request)
        return frame if frame is not None else self._zero_frame(request)

    @staticmethod
    def _zero_frame(request: InputConvFrameRequest) -> InputConvFrame:
        payload_bytes = expected_payload_bytes(
            request.width,
            request.height,
            request.channels,
            request.preferred_pixel_format,
        )
        return InputConvFrame(
            time_step=request.time_step,
            source_camera_index=request.source_camera_index,
            width=request.width,
            height=request.height,
            channels=request.channels,
            pixel_format=request.preferred_pixel_format,
            bytes=b"\x00" * payload_bytes,
        )

    @classmethod
    def _zero_frame_from_request_payload(cls, payload: bytes) -> InputConvFrame:
        try:
            return cls._zero_frame(unpack_input_conv_frame_request(payload))
        except Exception:
            return InputConvFrame(time_step=0, source_camera_index=0, width=1, height=1, channels=1, bytes=b"\x00")

    def summary(self) -> dict[str, int | str | bool | None]:
        return {
            "endpoint": self.endpoint,
            "bind_address": self.bind_address,
            "port": self.port,
            "closed": self.closed,
            "running": self.running,
            "received_requests": self._stats.received_requests,
            "sent_frames": self._stats.sent_frames,
            "provider_errors": self._stats.provider_errors,
            "last_request_time_step": self._stats.last_request_time_step,
            "last_inputconv_index": self._stats.last_inputconv_index,
            "last_error": self._stats.last_error,
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._closed = True


@dataclass(slots=True)
class InputConvFrameClientStats:
    requested_frames: int = 0
    received_frames: int = 0
    failed_requests: int = 0
    last_request_time_step: int | None = None
    last_received_time_step: int | None = None
    last_error: str = ""


class InputConvFrameClient:
    """Request synchronous InputConv frames from a REQ/REP frame server."""

    def __init__(
        self,
        *,
        connect_address: str = "127.0.0.1",
        port: int,
        linger_ms: int = 0,
        receive_timeout_ms: int = 2000,
        autostart: bool = True,
    ) -> None:
        self.connect_address = connect_address
        self.port = int(port)
        self.linger_ms = int(linger_ms)
        self.receive_timeout_ms = int(receive_timeout_ms)
        self._context: Any | None = None
        self._socket: Any | None = None
        self._closed = False
        self._stats = InputConvFrameClientStats()
        if autostart:
            self.start()

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.connect_address}:{self.port}"

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "InputConvFrameClient":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def start(self) -> "InputConvFrameClient":
        if self._socket is not None:
            return self
        zmq = _import_zmq()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.linger = self.linger_ms
        self._socket.rcvtimeo = self.receive_timeout_ms
        self._socket.connect(self.endpoint)
        self._closed = False
        return self

    def request_frame(self, request: InputConvFrameRequest) -> InputConvFrame:
        self.start()
        assert self._socket is not None
        self._stats.requested_frames += 1
        self._stats.last_request_time_step = request.time_step
        try:
            self._socket.send(pack_input_conv_frame_request(request))
            parts = self._socket.recv_multipart()
            frame = unpack_input_conv_frame(b"".join(parts))
            self._stats.received_frames += 1
            self._stats.last_received_time_step = frame.time_step
            self._stats.last_error = ""
            return frame
        except Exception as exc:
            self._stats.failed_requests += 1
            self._stats.last_error = str(exc)
            raise

    def summary(self) -> dict[str, int | str | bool | None]:
        return {
            "endpoint": self.endpoint,
            "connect_address": self.connect_address,
            "port": self.port,
            "closed": self.closed,
            "requested_frames": self._stats.requested_frames,
            "received_frames": self._stats.received_frames,
            "failed_requests": self._stats.failed_requests,
            "last_request_time_step": self._stats.last_request_time_step,
            "last_received_time_step": self._stats.last_received_time_step,
            "last_error": self._stats.last_error,
        }

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._closed = True
