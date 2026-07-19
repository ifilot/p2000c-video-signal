#!/usr/bin/env python3
"""Extract one centered P2000C monochrome video frame from a Siglent binary file.

Defaults are tuned for data/SDS00003.bin:

* CH1: horizontal sync
* CH2: monochrome video
* CH3: vertical sync
* CH4: additional video-level signal, probably at or near VID1
* 12.288 MHz dot clock, 98 character clocks per scanline
* 80 x 24 visible text area, 8 x 12 dots per character
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from video_capture import (
    TIMING_PROFILES,
    VISIBLE_LINES,
    channel_arg,
    infer_vsync_reference_edge,
    load_channels,
    resolve_channel,
    rising_edges,
    threshold_edges,
)


def extract_visible_frame(
    hsync: np.ndarray,
    video: np.ndarray,
    vsync: np.ndarray,
    sample_rate: float,
    *,
    full_lines: int,
    full_dots: int,
    visible_dots: int,
    visible_top_line: int,
    vsync_edge: str,
    hsync_threshold: float,
    vsync_threshold: float,
    active_start_dot: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Decode the visible 80x24-character raster from one VSYNC-bounded frame."""
    if not 0 <= visible_top_line <= full_lines - VISIBLE_LINES:
        raise ValueError(
            f"visible_top_line must fit {VISIBLE_LINES} visible lines inside "
            f"{full_lines} frame lines"
        )
    hsync_edges = rising_edges(hsync, hsync_threshold, min_gap=1000)
    if vsync_edge == "auto":
        vsync_edge = infer_vsync_reference_edge(vsync, vsync_threshold)
    vsync_edges = threshold_edges(
        vsync,
        vsync_threshold,
        min_gap=int(0.0001 * sample_rate),
        edge=vsync_edge,
    )
    if hsync_edges.size < full_lines:
        raise ValueError(f"Only found {hsync_edges.size} HSYNC edges")
    if vsync_edges.size < 2:
        raise ValueError(f"Need at least two VSYNC {vsync_edge} edges to extract one full frame")

    frame_start = int(vsync_edges[0])
    frame_stop = int(vsync_edges[1])
    # CH3 VSYNC rises shortly after a CH1 HSYNC edge, so the frame's scanline 0
    # is the HSYNC edge immediately preceding the VSYNC transition.
    line0_index = max(0, int(np.searchsorted(hsync_edges, frame_start, side="right") - 1))
    line_starts = hsync_edges[line0_index : line0_index + full_lines]
    if line_starts.size != full_lines:
        raise ValueError(f"Expected {full_lines} frame lines, got {line_starts.size}")

    # Use the measured line period instead of only the profile's nominal clock.
    # This removes sub-pixel drift across the visible dots when mapping
    # oscilloscope samples to video dots.
    line_period_samples = float(np.median(np.diff(hsync_edges)))
    samples_per_dot = line_period_samples / full_dots
    dot_centers = active_start_dot * samples_per_dot
    dot_centers += (np.arange(visible_dots, dtype=np.float64) + 0.5) * samples_per_dot
    # Average three nearby taps for each dot.  That keeps the binary image
    # robust against sampling phase and narrow analog spikes without blurring
    # across neighboring dot cells.
    taps = np.array([-0.25, 0.0, 0.25], dtype=np.float64) * samples_per_dot

    full_frame = np.empty((full_lines, visible_dots), dtype=np.float32)
    for row, line_start in enumerate(line_starts):
        samples = []
        for tap in taps:
            indices = np.rint(line_start + dot_centers + tap).astype(np.int64)
            indices = np.clip(indices, 0, video.size - 1)
            samples.append(video[indices])
        full_frame[row] = np.mean(samples, axis=0)

    # Drop the profile-specific non-visible scanlines before the 80x24 text area.
    visible = full_frame[visible_top_line : visible_top_line + VISIBLE_LINES]
    metadata = {
        "frame_start_sample": frame_start,
        "frame_stop_sample": frame_stop,
        "frame_duration_ms": (frame_stop - frame_start) / sample_rate * 1e3,
        "lines_per_frame": (frame_stop - frame_start) / line_period_samples,
        "first_hsync_sample": int(line_starts[0]),
        "line0_index": line0_index,
        "line_period_us": line_period_samples / sample_rate * 1e6,
        "measured_dot_clock_mhz": sample_rate / samples_per_dot / 1e6,
        "active_start_dot": active_start_dot,
        "full_lines": full_lines,
        "full_dots": full_dots,
        "visible_dots": visible_dots,
        "visible_top_line": visible_top_line,
        "vsync_reference_edge": vsync_edge,
        "visible_first_row": 1,
        "visible_last_row": 24,
    }
    return visible, metadata


def save_threshold_image(
    visible_frame: np.ndarray,
    output: Path,
    *,
    threshold: float,
    scale: int,
    invert: bool,
) -> None:
    """Threshold the decoded voltage raster and save a scaled monochrome PNG."""
    image = Image.fromarray((visible_frame > threshold).astype(np.uint8) * 255, mode="L")
    if invert:
        image = ImageOps.invert(image)
    image = image.resize(
        (visible_frame.shape[1] * scale, visible_frame.shape[0] * scale),
        Image.Resampling.NEAREST,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/SDS00003.bin"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/SDS00003_single_frame_centered_threshold_1.00.png"),
    )
    parser.add_argument(
        "--profile",
        choices=TIMING_PROFILES,
        default="p2000c",
        help="Timing defaults for the captured machine",
    )
    parser.add_argument("--threshold", type=float, default=1.0, help="CH2 dot threshold in volts")
    parser.add_argument("--hsync-threshold", type=float, default=2.0)
    parser.add_argument("--vsync-threshold", type=float, default=2.0)
    parser.add_argument(
        "--vsync-edge",
        choices=("profile", "auto", "rising", "falling"),
        default="profile",
        help="VSYNC edge used as the frame reference",
    )
    parser.add_argument("--hsync-channel", type=channel_arg, default=1)
    parser.add_argument("--video-channel", type=channel_arg, default=2)
    parser.add_argument("--vsync-channel", type=channel_arg, default=3)
    parser.add_argument("--visible-top-line", type=int)
    parser.add_argument(
        "--active-start-dot",
        type=int,
        help="First displayed dot relative to the CH1 HSYNC rising edge",
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    profile = TIMING_PROFILES[args.profile]
    vsync_edge = profile.vsync_reference_edge if args.vsync_edge == "profile" else args.vsync_edge
    visible_top_line = (
        profile.visible_top_line if args.visible_top_line is None else args.visible_top_line
    )
    active_start_dot = (
        profile.active_start_dot if args.active_start_dot is None else args.active_start_dot
    )
    channels, _, sample_rate = load_channels(args.input)
    visible_frame, metadata = extract_visible_frame(
        resolve_channel(channels, args.hsync_channel, "HSYNC"),
        resolve_channel(channels, args.video_channel, "video"),
        resolve_channel(channels, args.vsync_channel, "VSYNC"),
        sample_rate,
        full_lines=profile.full_lines,
        full_dots=profile.full_dots,
        visible_dots=profile.visible_dots,
        visible_top_line=visible_top_line,
        vsync_edge=vsync_edge,
        hsync_threshold=args.hsync_threshold,
        vsync_threshold=args.vsync_threshold,
        active_start_dot=active_start_dot,
    )
    save_threshold_image(
        visible_frame,
        args.output,
        threshold=args.threshold,
        scale=args.scale,
        invert=args.invert,
    )

    print(f"Wrote {args.output}")
    print(f"Profile: {profile.name}")
    print(f"Channels loaded: {len(channels)}")
    print(f"Using CH{args.hsync_channel}=HSYNC CH{args.video_channel}=video CH{args.vsync_channel}=VSYNC")
    print(f"Threshold: {args.threshold:g} V")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
