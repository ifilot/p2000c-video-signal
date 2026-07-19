#!/usr/bin/env python3
"""Render a full-capture raw waveform overview for the P2000C Siglent file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from video_capture import load_channels


CHANNEL_LABELS = (
    "CH1 HSYNC",
    "CH2 video",
    "CH3 VSYNC",
    "CH4 video level (VID1?)",
)
CHANNEL_COLORS = ("tab:blue", "tab:red", "tab:green", "tab:purple")


def waveform_envelope(
    signal: np.ndarray,
    sample_rate: float,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, minimum, and maximum arrays for a downsampled envelope."""
    if signal.size <= max_points:
        samples = np.arange(signal.size, dtype=np.float64)
        time_ms = (samples - signal.size / 2.0) / sample_rate * 1e3
        return time_ms, signal, signal

    bin_size = int(np.ceil(signal.size / max_points))
    bin_count = signal.size // bin_size
    trimmed = signal[: bin_count * bin_size].reshape(bin_count, bin_size)
    centers = (np.arange(bin_count, dtype=np.float64) + 0.5) * bin_size
    time_ms = (centers - signal.size / 2.0) / sample_rate * 1e3
    return time_ms, trimmed.min(axis=1), trimmed.max(axis=1)


def render_overview(
    channels: list[np.ndarray],
    sample_rate: float,
    input_path: Path,
    output_path: Path,
    *,
    max_points: int,
) -> None:
    """Render all channels as min/max envelopes over the full capture."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(channels),
        1,
        figsize=(15, 2.6 * len(channels)),
        sharex=True,
        constrained_layout=True,
    )
    if len(channels) == 1:
        axes = [axes]

    for index, (axis, signal) in enumerate(zip(axes, channels)):
        time_ms, low, high = waveform_envelope(signal, sample_rate, max_points=max_points)
        color = CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
        label = CHANNEL_LABELS[index] if index < len(CHANNEL_LABELS) else f"CH{index + 1}"
        axis.vlines(time_ms, low, high, color=color, linewidth=0.35)
        axis.plot(time_ms, (low + high) / 2.0, color=color, linewidth=0.35, alpha=0.55)
        axis.set_ylabel(f"{label}\nV")
        axis.grid(color="black", alpha=0.12, linewidth=0.5)

    axes[-1].set_xlabel("Time relative to capture center (ms)")
    fig.suptitle(f"Siglent {input_path.name}: raw waveform overview, full capture")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/SDS00003.bin"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/SDS00003_value_vs_time_overview.png"),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12_000,
        help="Maximum envelope bins to draw per channel",
    )
    args = parser.parse_args()

    channels, wave_length, sample_rate = load_channels(args.input)
    render_overview(
        channels,
        sample_rate,
        args.input,
        args.output,
        max_points=args.max_points,
    )

    print(f"Wrote {args.output}")
    print(f"Channels loaded: {len(channels)}")
    print(f"Wave length: {wave_length}")
    print(f"Sample rate: {sample_rate:g} Hz")


if __name__ == "__main__":
    main()
