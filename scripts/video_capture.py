"""Shared Siglent/P2000C video capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from pathlib import Path

import numpy as np


HEADER_SIZE = 2048

# Offsets in Siglent SDS binary files.  The scope stores one 2048-byte header
# followed by one contiguous unsigned-byte waveform block per enabled channel.
WAVE_LENGTH_OFFSET = 0x1E8
SAMPLE_RATE_OFFSET = 0x1EC
VDIV_OFFSETS = (0x14, 0x3C, 0x64, 0x8C)
VOLTAGE_OFFSET_OFFSETS = (0xB4, 0xDC, 0x104, 0x12C)
CODE_PER_DIV_OFFSETS = (0x26C, 0x270, 0x274, 0x278)

DOTS_PER_CHAR = 8
CHAR_ROWS_VISIBLE = 24
SCANLINES_PER_CHAR = 12

VISIBLE_LINES = CHAR_ROWS_VISIBLE * SCANLINES_PER_CHAR


@dataclass(frozen=True)
class TimingProfile:
    """Capture-specific timing choices for a P2000C video frame."""

    name: str
    title: str
    dot_clock_hz: float
    chars_per_line: int
    full_lines: int
    display_chars: int
    display_start_char: int
    active_start_dot: int
    visible_top_line: int
    vsync_reference_edge: str
    row_order_note: str

    @property
    def full_dots(self) -> int:
        """Total dot clocks in one scanline."""
        return self.chars_per_line * DOTS_PER_CHAR

    @property
    def visible_dots(self) -> int:
        """Visible text dots in one scanline."""
        return self.display_chars * DOTS_PER_CHAR

    @property
    def display_start_dot(self) -> int:
        """Visible display start in the profile's horizontal coordinate system."""
        return self.display_start_char * DOTS_PER_CHAR

    @property
    def display_end_dot(self) -> int:
        """Visible display end in the profile's horizontal coordinate system."""
        return self.display_start_dot + self.visible_dots


TIMING_PROFILES = {
    "p2000c": TimingProfile(
        name="p2000c",
        title="P2000C Video Timing: One VSYNC-Bounded Frame",
        dot_clock_hz=12_288_000.0,
        chars_per_line=98,
        full_lines=311,
        display_chars=80,
        display_start_char=17,
        active_start_dot=136,
        visible_top_line=16,
        vsync_reference_edge="rising",
        row_order_note=(
            "SDS00003 crop: 80x24 text starts 16 scanlines after CH3 rises; "
            "line timing is 98 character clocks at 12.288 MHz"
        ),
    ),
}


def read_u32(header: bytes, offset: int) -> int:
    """Read an unsigned 32-bit little-endian field from the Siglent header."""
    return struct.unpack_from("<I", header, offset)[0]


def read_f64(header: bytes, offset: int) -> float:
    """Read a 64-bit little-endian floating-point field from the Siglent header."""
    return struct.unpack_from("<d", header, offset)[0]


def load_channels(path: Path) -> tuple[list[np.ndarray], int, float]:
    """Load enabled channels and scale their raw ADC bytes into volts."""
    with path.open("rb") as handle:
        header = handle.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise ValueError(f"{path} is too small to contain a Siglent header")

    wave_length = read_u32(header, WAVE_LENGTH_OFFSET)
    sample_rate = read_f64(header, SAMPLE_RATE_OFFSET)
    if wave_length <= 0:
        raise ValueError(f"{path} has an invalid waveform length: {wave_length}")

    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_SIZE)
    channel_count, remainder = divmod(raw.size, wave_length)
    if remainder:
        raise ValueError(
            f"File has {raw.size} data bytes, which is not a whole number of "
            f"{wave_length}-sample channels"
        )
    if channel_count < 3:
        raise ValueError(f"Need at least CH1/CH2/CH3, found {channel_count} channels")
    if channel_count > len(VDIV_OFFSETS):
        raise ValueError(f"Only up to 4 Siglent channels are supported, found {channel_count}")

    channels: list[np.ndarray] = []
    for ch in range(channel_count):
        # Siglent's byte code 128 represents the center of the vertical scale;
        # vdiv, offset, and codes-per-division convert ADC codes back to volts.
        channel_raw = np.asarray(
            raw[ch * wave_length : (ch + 1) * wave_length], dtype=np.float32
        )
        vdiv = read_f64(header, VDIV_OFFSETS[ch])
        offset = read_f64(header, VOLTAGE_OFFSET_OFFSETS[ch])
        codes_per_div = read_u32(header, CODE_PER_DIV_OFFSETS[ch])
        volts = (channel_raw - 128.0) * vdiv / codes_per_div + offset
        channels.append(volts)
    return channels, wave_length, sample_rate


def threshold_edges(
    signal: np.ndarray,
    threshold: float,
    min_gap: int,
    *,
    edge: str,
) -> np.ndarray:
    """Return threshold edges, suppressing duplicate/noisy crossings."""
    high = signal > threshold
    if edge == "rising":
        edges = np.flatnonzero(~high[:-1] & high[1:]) + 1
    elif edge == "falling":
        edges = np.flatnonzero(high[:-1] & ~high[1:]) + 1
    else:
        raise ValueError(f"Unsupported edge: {edge}")
    if edges.size == 0:
        return edges

    clean = [int(edges[0])]
    for crossing in edges[1:]:
        if int(crossing) - clean[-1] > min_gap:
            clean.append(int(crossing))
    return np.array(clean, dtype=np.int64)


def rising_edges(signal: np.ndarray, threshold: float, min_gap: int) -> np.ndarray:
    """Return threshold rising edges, suppressing duplicate/noisy crossings."""
    return threshold_edges(signal, threshold, min_gap, edge="rising")


def falling_edges(signal: np.ndarray, threshold: float, min_gap: int) -> np.ndarray:
    """Return threshold falling edges, suppressing duplicate/noisy crossings."""
    return threshold_edges(signal, threshold, min_gap, edge="falling")


def infer_vsync_reference_edge(signal: np.ndarray, threshold: float) -> str:
    """Choose the likely beginning of a VSYNC pulse from signal duty cycle."""
    # SDS00003's CH3 is high most of the time with low-going VSYNC pulses.
    return "falling" if float(np.mean(signal > threshold)) > 0.5 else "rising"


def channel_arg(value: str) -> int:
    """Parse a one-based channel number for argparse."""
    channel = int(value)
    if channel < 1:
        raise ValueError("channel numbers are one-based and must be >= 1")
    return channel


def resolve_channel(channels: list[np.ndarray], channel: int, role: str) -> np.ndarray:
    """Return a one-based channel or raise a friendly error."""
    if channel > len(channels):
        raise ValueError(f"{role} uses CH{channel}, but the capture only has {len(channels)} channels")
    return channels[channel - 1]
