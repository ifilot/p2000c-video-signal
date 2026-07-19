#!/usr/bin/env python3
"""Render a one-frame P2000C video timing/synthesis diagram.

The diagram uses the measured CH1/CH2/CH3 capture plus the selected profile's
dot timing:

* top timing lanes: CH1 horizontal sync and the manual's display-enable window
* left timing lane: CH3 vertical sync
* main raster: display area, 8 x 12 character grid, and CH2 video dots

Defaults are tuned for data/SDS00003.bin.  The profile uses the manual's
12.288 MHz dot clock and 98 character clocks per scanline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from video_capture import (
    DOTS_PER_CHAR,
    SCANLINES_PER_CHAR,
    TIMING_PROFILES,
    VISIBLE_LINES,
    channel_arg,
    falling_edges,
    infer_vsync_reference_edge,
    load_channels,
    resolve_channel,
    rising_edges,
    threshold_edges,
)


def build_diagram(
    hsync: np.ndarray,
    video: np.ndarray,
    vsync: np.ndarray,
    sample_rate: float,
    *,
    full_lines: int,
    full_dots: int,
    display_start_dot: int,
    display_end_dot: int,
    visible_top_line: int,
    vsync_edge: str,
    hsync_threshold: float,
    vsync_threshold: float,
    video_threshold: float,
    active_start_dot: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Build an RGB raster showing sync timing, video enable, and lit pixels."""
    if not 0 <= visible_top_line <= full_lines - VISIBLE_LINES:
        raise ValueError(
            f"visible_top_line must fit {VISIBLE_LINES} visible lines inside "
            f"{full_lines} frame lines"
        )
    hsync_edges = rising_edges(hsync, hsync_threshold, min_gap=1000)
    hsync_falls = falling_edges(hsync, hsync_threshold, min_gap=1000)
    if vsync_edge == "auto":
        vsync_edge = infer_vsync_reference_edge(vsync, vsync_threshold)
    vsync_edges = threshold_edges(
        vsync,
        vsync_threshold,
        min_gap=int(0.0001 * sample_rate),
        edge=vsync_edge,
    )
    opposite_vsync_edges = threshold_edges(
        vsync,
        vsync_threshold,
        min_gap=int(0.0001 * sample_rate),
        edge="falling" if vsync_edge == "rising" else "rising",
    )
    if vsync_edges.size < 2:
        raise ValueError(f"Need at least two VSYNC {vsync_edge} edges to render one full frame")

    frame_start = int(vsync_edges[0])
    frame_stop = int(vsync_edges[1])
    # The vertical sync transition occurs after the horizontal sync edge that
    # starts the first line of the frame, so use the preceding CH1 edge as line 0.
    line0_index = max(0, int(np.searchsorted(hsync_edges, frame_start, side="right") - 1))
    line_starts = hsync_edges[line0_index : line0_index + full_lines]
    if line_starts.size != full_lines:
        raise ValueError(f"Expected {full_lines} frame lines, got {line_starts.size}")

    # Base dot positions on the captured line period, not just the nominal
    # profile's nominal clock.  This keeps the final dots aligned with the
    # measured signal over the full scanline.
    line_period_samples = float(np.median(np.diff(hsync_edges)))
    samples_per_dot = line_period_samples / full_dots
    dot_centers = (np.arange(full_dots, dtype=np.float64) + 0.5) * samples_per_dot

    manual_origin_after_hsync_edge = active_start_dot - display_start_dot
    hsync_edge_manual_dot = (-manual_origin_after_hsync_edge) % full_dots

    # Measure the CH1 pulse width from nearby falling edges and display the
    # result in profile coordinates.
    hsync_width_samples = []
    for edge in hsync_edges[: min(20, hsync_edges.size)]:
        later_falls = hsync_falls[hsync_falls > edge]
        if later_falls.size:
            width = int(later_falls[0] - edge)
            if width < int(0.5 * np.median(np.diff(hsync_edges))):
                hsync_width_samples.append(width)
    hsync_width_dots = float(np.median(hsync_width_samples) / samples_per_dot)
    if opposite_vsync_edges.size:
        nearest_opposite = int(
            opposite_vsync_edges[np.argmin(np.abs(opposite_vsync_edges - frame_start))]
        )
        vsync_width_lines = float(abs(nearest_opposite - frame_start) / line_period_samples)
    else:
        vsync_width_lines = 0.0

    # Start with a black frame, then paint the display-enable region
    # dark green.  Actual CH2-lit dots will be painted white on top.
    rgb = np.zeros((full_lines, full_dots, 3), dtype=np.float32)
    rgb[
        visible_top_line : visible_top_line + VISIBLE_LINES,
        display_start_dot:display_end_dot,
        1,
    ] = 0.16

    for row, line_start in enumerate(line_starts):
        # Shift sampling into the profile coordinate system so the oscilloscope
        # waveform, service-manual timing, and rendered grid share one
        # horizontal origin.
        indices = np.rint(
            line_start + manual_origin_after_hsync_edge * samples_per_dot + dot_centers
        ).astype(np.int64)
        indices = np.clip(indices, 0, video.size - 1)

        video_high = video[indices] > video_threshold

        active_video = np.zeros(full_dots, dtype=bool)
        if visible_top_line <= row < visible_top_line + VISIBLE_LINES:
            # CH2 is only meaningful inside the 80-character display-enable
            # area.  Pixels outside that area are timing/blanking space.
            active_video[display_start_dot:display_end_dot] = video_high[
                display_start_dot:display_end_dot
            ]
        rgb[row, active_video] = (1.0, 1.0, 1.0)

    # Character grid in the displayed area, useful for seeing the 8-dot/12-line structure.
    for col in range(display_start_dot, display_end_dot + 1, DOTS_PER_CHAR):
        if 0 <= col < full_dots:
            rgb[visible_top_line : visible_top_line + VISIBLE_LINES, col, 1] = np.maximum(
                rgb[visible_top_line : visible_top_line + VISIBLE_LINES, col, 1], 0.45
            )
    for row in range(
        visible_top_line,
        visible_top_line + VISIBLE_LINES,
        SCANLINES_PER_CHAR,
    ):
        rgb[row, display_start_dot:display_end_dot, 1] = np.maximum(
            rgb[row, display_start_dot:display_end_dot, 1], 0.45
        )

    metadata = {
        "frame_start_sample": frame_start,
        "frame_stop_sample": frame_stop,
        "frame_duration_ms": (frame_stop - frame_start) / sample_rate * 1e3,
        "lines_per_frame": (frame_stop - frame_start) / line_period_samples,
        "line0_index": line0_index,
        "line_period_us": line_period_samples / sample_rate * 1e6,
        "measured_dot_clock_mhz": sample_rate / samples_per_dot / 1e6,
        "active_start_dot": active_start_dot,
        "full_lines": full_lines,
        "full_dots": full_dots,
        "display_start_dot_manual": display_start_dot,
        "display_end_dot_manual": display_end_dot,
        "display_start_line": visible_top_line,
        "display_end_line": visible_top_line + VISIBLE_LINES,
        "manual_origin_after_hsync_edge": manual_origin_after_hsync_edge,
        "hsync_edge_manual_dot": int(hsync_edge_manual_dot),
        "hsync_width_dots": hsync_width_dots,
        "vsync_width_lines": vsync_width_lines,
        "vsync_reference_edge": vsync_edge,
        "video_threshold": video_threshold,
    }
    return rgb, metadata


def save_diagram(
    rgb: np.ndarray,
    output: Path,
    metadata: dict[str, float | int | str],
    *,
    title: str,
    row_order_note: str,
) -> None:
    """Lay out the timing lanes and raster into a single explanatory PNG."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=(1.0, 7.0),
        width_ratios=(0.45, 12.0),
    )
    fig.suptitle(title)

    # Use three axes rather than drawing everything into one image: the top axis
    # explains horizontal timing, the left axis explains vertical timing, and
    # the large axis shows the synthesized screen raster.
    ax_top = fig.add_subplot(grid[0, 1])
    ax_left = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[1, 1])
    full_dots = int(metadata["full_dots"])
    full_lines = int(metadata["full_lines"])
    display_start_dot = int(metadata["display_start_dot_manual"])
    display_end_dot = int(metadata["display_end_dot_manual"])
    visible_dots = display_end_dot - display_start_dot
    display_start_char = display_start_dot // DOTS_PER_CHAR
    display_end_char = display_end_dot // DOTS_PER_CHAR

    # Top lanes in the profile's character-counter coordinate.
    ax_top.set_xlim(0, full_dots)
    ax_top.set_ylim(0, 2)
    ax_top.set_yticks([0.5, 1.5])
    ax_top.set_yticklabels(["VIDEO ENABLE", "CH1 HSYNC"])
    ax_top.set_xticks(range(0, full_dots + 1, 8 * 12))
    ax_top.grid(color="black", alpha=0.12, linewidth=0.5)
    ax_top.add_patch(
        Rectangle(
            (display_start_dot, 0.15),
            visible_dots,
            0.7,
            facecolor=(0.0, 0.55, 0.0),
            edgecolor="none",
        )
    )

    hsync_start = int(metadata["hsync_edge_manual_dot"])
    hsync_width = int(round(float(metadata["hsync_width_dots"])))
    first_width = min(full_dots - hsync_start, hsync_width)
    # Horizontal sync can wrap around the profile coordinate origin, so it may
    # need two rectangles.
    ax_top.add_patch(
        Rectangle((hsync_start, 1.15), first_width, 0.7, facecolor="red", edgecolor="none")
    )
    if hsync_width > first_width:
        ax_top.add_patch(
            Rectangle(
                (0, 1.15),
                hsync_width - first_width,
                0.7,
                facecolor="red",
                edgecolor="none",
            )
        )
    ax_top.set_title("Horizontal timing lanes in manual dot coordinates")
    ax_top.set_xlabel("Dot position / character-counter position")

    # Left lane for the measured vertical sync interval.
    ax_left.set_ylim(full_lines, 0)
    ax_left.set_xlim(0, 1)
    ax_left.set_xticks([])
    ax_left.set_yticks(range(0, full_lines + 1, 24))
    ax_left.set_ylabel("Scanline within frame")
    vsync_rows = int(round(float(metadata["vsync_width_lines"])))
    ax_left.add_patch(
        Rectangle((0.15, 0), 0.7, vsync_rows, facecolor="blue", edgecolor="none")
    )
    ax_left.axhline(SCANLINES_PER_CHAR, color="black", alpha=0.35, linewidth=0.8)
    ax_left.axhline(
        int(metadata["display_start_line"]),
        color="black",
        alpha=0.35,
        linewidth=0.8,
    )
    ax_left.set_title("CH3\nVSYNC")

    # The main raster is already in manual dot/scanline coordinates: green is
    # the display-enable area, stronger green lines are character-cell guides,
    # and white dots are CH2 samples above the requested video threshold.
    ax.imshow(rgb, interpolation="nearest", aspect="auto")
    ax.set_xlabel("Dot position within one scanline")
    ax.set_xticks(range(0, full_dots + 1, 8 * 12))
    ax.set_yticks(range(0, full_lines + 1, 24))
    ax.grid(color="white", alpha=0.12, linewidth=0.5)
    ax.legend(
        handles=[
            Patch(facecolor="red", label="CH1 HSYNC"),
            Patch(facecolor="blue", label="CH3 VSYNC"),
            Patch(
                facecolor=(0.0, 0.45, 0.0),
                label=f"{visible_dots // DOTS_PER_CHAR} x 24 video window / 8 x 12 grid",
            ),
            Patch(facecolor="white", edgecolor="black", label="CH2 video dot"),
        ],
        loc="upper right",
        framealpha=0.9,
    )
    note = (
        f"Frame {metadata['frame_duration_ms']:.3f} ms, "
        f"line {metadata['line_period_us']:.3f} us, "
        f"active start dot {metadata['active_start_dot']}, "
        f"display C{display_start_char:02d}-C{display_end_char:02d}, "
        f"CH1 edge at C{metadata['hsync_edge_manual_dot'] // DOTS_PER_CHAR:02.0f}, "
        f"video threshold {metadata['video_threshold']:g} V, "
        f"VSYNC {metadata['vsync_reference_edge']} reference\n"
        f"{row_order_note}"
    )
    ax.text(
        0.01,
        -0.09,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/SDS00003.bin"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/SDS00003_video_timing_synthesis_diagram.png"),
    )
    parser.add_argument(
        "--profile",
        choices=TIMING_PROFILES,
        default="p2000c",
        help="Timing defaults for the captured machine",
    )
    parser.add_argument("--video-threshold", type=float, default=1.0)
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
    parser.add_argument("--active-start-dot", type=int)
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
    rgb, metadata = build_diagram(
        resolve_channel(channels, args.hsync_channel, "HSYNC"),
        resolve_channel(channels, args.video_channel, "video"),
        resolve_channel(channels, args.vsync_channel, "VSYNC"),
        sample_rate,
        full_lines=profile.full_lines,
        full_dots=profile.full_dots,
        display_start_dot=profile.display_start_dot,
        display_end_dot=profile.display_end_dot,
        visible_top_line=visible_top_line,
        vsync_edge=vsync_edge,
        hsync_threshold=args.hsync_threshold,
        vsync_threshold=args.vsync_threshold,
        video_threshold=args.video_threshold,
        active_start_dot=active_start_dot,
    )
    save_diagram(
        rgb,
        args.output,
        metadata,
        title=profile.title,
        row_order_note=profile.row_order_note,
    )

    print(f"Wrote {args.output}")
    print(f"Profile: {profile.name}")
    print(f"Channels loaded: {len(channels)}")
    print(f"Using CH{args.hsync_channel}=HSYNC CH{args.video_channel}=video CH{args.vsync_channel}=VSYNC")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
