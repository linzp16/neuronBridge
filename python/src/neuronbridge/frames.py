"""Dynamic InputConv frame helpers and wire protocol utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct


class InputConvPixelFormat(IntEnum):
    UINT8_GRAY = 0
    UINT8_RGB = 1
    UINT8_BGR = 2
    FLOAT32_GRAY = 3
    FLOAT32_HWC = 4
    FLOAT32_CHW = 5

    @property
    def native_name(self) -> str:
        return {
            InputConvPixelFormat.UINT8_GRAY: "uint8_gray",
            InputConvPixelFormat.UINT8_RGB: "uint8_rgb",
            InputConvPixelFormat.UINT8_BGR: "uint8_bgr",
            InputConvPixelFormat.FLOAT32_GRAY: "float32_gray",
            InputConvPixelFormat.FLOAT32_HWC: "float32_hwc",
            InputConvPixelFormat.FLOAT32_CHW: "float32_chw",
        }[self]


_PIXEL_FORMAT_BY_NAME = {
    "uint8_gray": InputConvPixelFormat.UINT8_GRAY,
    "gray": InputConvPixelFormat.UINT8_GRAY,
    "uint8_rgb": InputConvPixelFormat.UINT8_RGB,
    "rgb": InputConvPixelFormat.UINT8_RGB,
    "uint8_bgr": InputConvPixelFormat.UINT8_BGR,
    "bgr": InputConvPixelFormat.UINT8_BGR,
    "float32_gray": InputConvPixelFormat.FLOAT32_GRAY,
    "float32_hwc": InputConvPixelFormat.FLOAT32_HWC,
    "float32_chw": InputConvPixelFormat.FLOAT32_CHW,
}

SYNC_REQUEST_MAGIC = 0x49434652
SYNC_RESPONSE_MAGIC = 0x49434650
ASYNC_FRAME_MAGIC = 0x41494346
ASYNC_FRAME_VERSION = 1

SYNC_REQUEST_STRUCT = struct.Struct("<Iiiiiiiii")
SYNC_RESPONSE_STRUCT = struct.Struct("<IiiiiiiI")
ASYNC_FRAME_STRUCT = struct.Struct("<IIiiiiiiI")


def input_conv_pixel_format(value: InputConvPixelFormat | str | int) -> InputConvPixelFormat:
    if isinstance(value, InputConvPixelFormat):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _PIXEL_FORMAT_BY_NAME:
            raise ValueError(f"unsupported InputConv pixel format: {value}")
        return _PIXEL_FORMAT_BY_NAME[key]
    return InputConvPixelFormat(int(value))


@dataclass(slots=True)
class InputConvFrame:
    time_step: int
    source_camera_index: int
    width: int
    height: int
    channels: int
    bytes: bytes | bytearray | memoryview
    pixel_format: InputConvPixelFormat | str | int = InputConvPixelFormat.UINT8_GRAY

    def __post_init__(self) -> None:
        self.time_step = int(self.time_step)
        self.source_camera_index = int(self.source_camera_index)
        self.width = int(self.width)
        self.height = int(self.height)
        self.channels = int(self.channels)
        self.pixel_format = input_conv_pixel_format(self.pixel_format)
        self.bytes = bytes(self.bytes)
        if self.width <= 0 or self.height <= 0 or self.channels <= 0:
            raise ValueError("InputConvFrame width, height, and channels must be positive")
        expected = expected_payload_bytes(self.width, self.height, self.channels, self.pixel_format)
        if len(self.bytes) != expected:
            raise ValueError(f"InputConvFrame payload has {len(self.bytes)} bytes; expected {expected}")

    def to_native(self) -> dict:
        return {
            "time_step": self.time_step,
            "source_camera_index": self.source_camera_index,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "pixel_format": self.pixel_format.native_name,
            "bytes": self.bytes,
        }


@dataclass(slots=True)
class InputConvFrameRequest:
    time_step: int
    inputconv_index: int
    source_camera_index: int
    width: int
    height: int
    channels: int
    preferred_pixel_format: InputConvPixelFormat | str | int = InputConvPixelFormat.UINT8_GRAY
    max_lag_steps: int = 0

    def __post_init__(self) -> None:
        self.time_step = int(self.time_step)
        self.inputconv_index = int(self.inputconv_index)
        self.source_camera_index = int(self.source_camera_index)
        self.width = int(self.width)
        self.height = int(self.height)
        self.channels = int(self.channels)
        self.preferred_pixel_format = input_conv_pixel_format(self.preferred_pixel_format)
        self.max_lag_steps = int(self.max_lag_steps)


def expected_payload_bytes(width: int, height: int, channels: int, pixel_format: InputConvPixelFormat | str | int) -> int:
    fmt = input_conv_pixel_format(pixel_format)
    if fmt in {InputConvPixelFormat.UINT8_GRAY, InputConvPixelFormat.FLOAT32_GRAY}:
        logical_channels = 1
    elif fmt in {InputConvPixelFormat.UINT8_RGB, InputConvPixelFormat.UINT8_BGR}:
        logical_channels = 3
    else:
        logical_channels = int(channels)
    bytes_per_value = 4 if fmt in {
        InputConvPixelFormat.FLOAT32_GRAY,
        InputConvPixelFormat.FLOAT32_HWC,
        InputConvPixelFormat.FLOAT32_CHW,
    } else 1
    return int(width) * int(height) * logical_channels * bytes_per_value


def pack_input_conv_frame_request(request: InputConvFrameRequest) -> bytes:
    return SYNC_REQUEST_STRUCT.pack(
        SYNC_REQUEST_MAGIC,
        request.time_step,
        request.inputconv_index,
        request.source_camera_index,
        request.width,
        request.height,
        request.channels,
        int(request.preferred_pixel_format),
        request.max_lag_steps,
    )


def unpack_input_conv_frame_request(payload: bytes) -> InputConvFrameRequest:
    values = SYNC_REQUEST_STRUCT.unpack(payload)
    if values[0] != SYNC_REQUEST_MAGIC:
        raise ValueError("invalid InputConv frame request magic")
    return InputConvFrameRequest(
        time_step=values[1],
        inputconv_index=values[2],
        source_camera_index=values[3],
        width=values[4],
        height=values[5],
        channels=values[6],
        preferred_pixel_format=values[7],
        max_lag_steps=values[8],
    )


def pack_input_conv_frame(frame: InputConvFrame, *, async_message: bool = False) -> bytes:
    if async_message:
        header = ASYNC_FRAME_STRUCT.pack(
            ASYNC_FRAME_MAGIC,
            ASYNC_FRAME_VERSION,
            frame.time_step,
            frame.source_camera_index,
            frame.width,
            frame.height,
            frame.channels,
            int(frame.pixel_format),
            len(frame.bytes),
        )
    else:
        header = SYNC_RESPONSE_STRUCT.pack(
            SYNC_RESPONSE_MAGIC,
            frame.time_step,
            frame.source_camera_index,
            frame.width,
            frame.height,
            frame.channels,
            int(frame.pixel_format),
            len(frame.bytes),
        )
    return header + frame.bytes


def unpack_input_conv_frame(payload: bytes, *, async_message: bool = False) -> InputConvFrame:
    if async_message:
        header_size = ASYNC_FRAME_STRUCT.size
        values = ASYNC_FRAME_STRUCT.unpack(payload[:header_size])
        if values[0] != ASYNC_FRAME_MAGIC or values[1] != ASYNC_FRAME_VERSION:
            raise ValueError("invalid async InputConv frame header")
        offset = 2
    else:
        header_size = SYNC_RESPONSE_STRUCT.size
        values = SYNC_RESPONSE_STRUCT.unpack(payload[:header_size])
        if values[0] != SYNC_RESPONSE_MAGIC:
            raise ValueError("invalid InputConv frame response magic")
        offset = 1
    payload_bytes = int(values[offset + 6])
    frame_payload = payload[header_size:header_size + payload_bytes]
    if len(frame_payload) != payload_bytes:
        raise ValueError("incomplete InputConv frame payload")
    return InputConvFrame(
        time_step=values[offset],
        source_camera_index=values[offset + 1],
        width=values[offset + 2],
        height=values[offset + 3],
        channels=values[offset + 4],
        pixel_format=values[offset + 5],
        bytes=frame_payload,
    )
