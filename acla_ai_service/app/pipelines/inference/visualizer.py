"""Utilities to render telemetry segment visualizations without coupling to learning services."""

from __future__ import annotations

import base64
import dataclasses
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.domain.expert_features import ExpertFeatureCatalog
from app.domain.tire_grip_features import TireGripFeatureCatalog


def visualize_optimal_segments(
    optimal_segments: List[List[Dict[str, Any]]],
    *,
    analyze_segment_fn: Optional[Callable[[pd.DataFrame, Sequence[str]], Dict[str, Any]]] = None,
    max_segments: int = 3,
    random_seed: Optional[int] = None,
    show: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    return_base64: bool = True,
    file_name_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create trajectory and speed visualizations for optimal telemetry segments.

    Args:
        optimal_segments: Segments returned from filter_optimal_telemetry_segments.
        analyze_segment_fn: Optional callable to calculate summary metrics per segment.
    max_segments: Maximum number of segments to visualize.
    random_seed: Optional seed for reproducible random sampling of segments.
        show: Display figures interactively when True.
        output_dir: Directory to persist plot images as .png files.
        return_base64: Include base64 PNG payloads in the response when True.
        file_name_prefix: Prefix for saved figure filenames.

    Returns:
        List of dictionaries containing metadata for each rendered figure.
    """

    if not optimal_segments:
        return []

    eo = ExpertFeatureCatalog.ExpertOptimalFeature
    context = ExpertFeatureCatalog.ContextFeature
    tire_context = TireGripFeatureCatalog.ContextFeature

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "output"
            / "transformer_eval"
            / "figures"
        )

    output_path: Optional[Path] = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    visualization_payloads: List[Dict[str, Any]] = []
    expert_context_features = [
        context.EXPERT_VELOCITY_ALIGNMENT.value,
        context.SPEED_DIFFERENCE.value,
        context.DISTANCE_TO_EXPERT_LINE.value,
        context.EXPERT_TIME_DIFFERENCE.value,
    ]
    tire_context_features = [
        tire_context.DRIVER_PUSH_TO_LIMIT.value,
    ]

    required_columns = [
        "Graphics_player_pos_x",
        "Graphics_player_pos_y",
        "Graphics_player_pos_z",
        "Graphics_current_time",
        "Physics_velocity_x",
        "Physics_velocity_y",
        "Physics_velocity_z",
        "Physics_speed_kmh",
        "Physics_gas",
        "Physics_brake",
        "Physics_steer_angle",
        eo.EXPERT_OPTIMAL_PLAYER_POS_X.value,
        eo.EXPERT_OPTIMAL_PLAYER_POS_Y.value,
        eo.EXPERT_OPTIMAL_PLAYER_POS_Z.value,
        eo.EXPERT_OPTIMAL_VELOCITY_X.value,
        eo.EXPERT_OPTIMAL_VELOCITY_Y.value,
        eo.EXPERT_OPTIMAL_VELOCITY_Z.value,
        eo.EXPERT_OPTIMAL_SPEED.value,
        *expert_context_features,
        *tire_context_features,
    ]

    segment_count = min(max_segments, len(optimal_segments))
    if segment_count <= 0:
        return []

    rng = np.random.default_rng(random_seed)
    selected_indices = sorted(
        rng.choice(len(optimal_segments), size=segment_count, replace=False).tolist()
    )

    for segment_idx in selected_indices:
        idx = int(segment_idx)
        segment_records = optimal_segments[idx]
        if not segment_records:
            continue

        segment_df = pd.DataFrame(segment_records).reset_index(drop=True)

        missing_columns = [col for col in required_columns if col not in segment_df.columns]
        if missing_columns:
            raise ValueError(
                f"Segment {idx + 1} missing required telemetry features: {', '.join(missing_columns)}"
            )

        current_x = segment_df["Graphics_player_pos_x"]
        current_y = segment_df["Graphics_player_pos_y"]
        current_z = segment_df["Graphics_player_pos_z"]

        expert_x = segment_df[eo.EXPERT_OPTIMAL_PLAYER_POS_X.value]
        expert_y = segment_df[eo.EXPERT_OPTIMAL_PLAYER_POS_Y.value]
        expert_z = segment_df[eo.EXPERT_OPTIMAL_PLAYER_POS_Z.value]

        current_speed_series = segment_df["Physics_speed_kmh"]
        expert_speed_series = segment_df[eo.EXPERT_OPTIMAL_SPEED.value]

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.0], width_ratios=[1.5, 1.0])
        ax_track = fig.add_subplot(gs[:, 0], projection="3d")
        ax_speed = fig.add_subplot(gs[0, 1])
        ax_time_delta = fig.add_subplot(gs[1, 1], sharex=ax_speed)
        ax_push_index = fig.add_subplot(gs[2, 1], sharex=ax_speed)

        current_x_np = current_x.to_numpy()
        current_y_np = current_y.to_numpy()
        current_z_np = current_z.to_numpy()
        ax_track.plot(current_x_np, current_y_np, current_z_np, color="#1f77b4", linewidth=2, label="Driver trajectory")
        ax_track.scatter(current_x_np[0], current_y_np[0], current_z_np[0], color="green", label="Segment start", s=50, depthshade=False)
        ax_track.scatter(current_x_np[-1], current_y_np[-1], current_z_np[-1], color="red", label="Segment end", s=50, depthshade=False)

        expert_x_np = expert_x.to_numpy()
        expert_y_np = expert_y.to_numpy()
        expert_z_np = expert_z.to_numpy()
        ax_track.plot(expert_x_np, expert_y_np, expert_z_np, color="#ff7f0e", linewidth=2, linestyle="--", label="Expert trajectory")

        ax_track.set_title(f"Segment {idx + 1} trajectory (3D)")
        ax_track.set_xlabel("Track X position")
        ax_track.set_ylabel("Track Y position")
        ax_track.set_zlabel("Track Z position")
        ax_track.grid(True, linestyle="--", alpha=0.2)

        try:
            ranges = np.array([
                np.ptp(current_x_np),
                np.ptp(current_y_np),
                np.ptp(current_z_np),
            ])
            ranges[ranges == 0] = 1.0
            ax_track.set_box_aspect(ranges)
        except Exception:
            pass

        ax_track.view_init(elev=20, azim=-60)
        ax_track.legend(loc="upper left")

        # Use telemetry timestamp (ms) converted to seconds for the x-axis
        segment_times_seconds = segment_df["Graphics_current_time"].astype(float).to_numpy() / 1000.0
        legend_handles: List[Any] = []
        legend_labels: List[str] = []

        driver_speed_line, = ax_speed.plot(
            segment_times_seconds,
            current_speed_series.to_numpy(),
            label="Driver speed (km/h)",
            color="#1f77b4",
            linewidth=2,
        )
        legend_handles.append(driver_speed_line)
        legend_labels.append("Driver speed (km/h)")

        expert_speed_line, = ax_speed.plot(
            segment_times_seconds,
            expert_speed_series.to_numpy(),
            label="Expert speed (km/h)",
            color="#ff7f0e",
            linestyle="--",
            linewidth=2,
        )
        legend_handles.append(expert_speed_line)
        legend_labels.append("Expert speed (km/h)")
        ax_speed.set_title("Speed profile across segment")
        ax_speed.set_xlabel("Segment time (s)")
        ax_speed.set_ylabel("Speed (km/h)")
        ax_speed.grid(True, linestyle="--", alpha=0.3)

        ax_speed2 = ax_speed.twinx()
        if context.EXPERT_VELOCITY_ALIGNMENT.value in segment_df:
            alignment_line, = ax_speed2.plot(
                segment_times_seconds,
                segment_df[context.EXPERT_VELOCITY_ALIGNMENT.value],
                color="purple",
                alpha=0.5,
                label="Velocity alignment",
            )
            ax_speed2.set_ylabel("Alignment (cosine)")
            legend_handles.append(alignment_line)
            legend_labels.append("Velocity alignment")
        ax_speed2.set_ylim(-1.05, 1.05)

        if legend_handles:
            ax_speed.legend(legend_handles, legend_labels, loc="lower left")

        # Derive sample deltas from telemetry timestamps since explicit deltas are no longer provided.
        telemetry_times_ms = segment_df["Graphics_current_time"].astype(float).to_numpy()
        
        # Check if we have expert time difference
        if context.EXPERT_TIME_DIFFERENCE.value in segment_df:
            # Time difference is in ms (from calculation), convert to seconds
            time_diff_series = segment_df[context.EXPERT_TIME_DIFFERENCE.value].to_numpy(dtype=float) / 1000.0
            
            ax_time_delta.plot(
                segment_times_seconds,
                time_diff_series,
                color="#e377c2", # Pink/Magenta
                linewidth=2,
                label="Time Delta to Expert",
            )
            ax_time_delta.set_ylabel("Delta to Expert (s)")
            ax_time_delta.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            # Add legend
            ax_time_delta.legend(loc="upper right")
            
        else:
            # Fallback: Time delta between samples (sampling rate)
            time_delta_series = np.diff(telemetry_times_ms, prepend=telemetry_times_ms[0]) / 1000.0
            ax_time_delta.plot(
                segment_times_seconds,
                time_delta_series,
                color="#2ca02c",
                linewidth=2,
            )
            ax_time_delta.set_ylabel("Sample Delta t (seconds)")
            
        ax_time_delta.set_title("Time Analysis")
        ax_time_delta.set_xlabel("")
        ax_time_delta.grid(True, linestyle="--", alpha=0.3)
        ax_speed.tick_params(labelbottom=True)
        ax_time_delta.tick_params(labelbottom=False)

        push_to_limit_series = segment_df[
            tire_context.DRIVER_PUSH_TO_LIMIT.value
        ].to_numpy(dtype=float)

        gas_series = segment_df["Physics_gas"].to_numpy(dtype=float)
        brake_series = segment_df["Physics_brake"].to_numpy(dtype=float)
        steer_series = segment_df["Physics_steer_angle"].to_numpy(dtype=float)

        ax_push_index.plot(
            segment_times_seconds,
            push_to_limit_series,
            color="#d62728",
            linewidth=2,
            label="Push-to-limit",
        )
        ax_push_index.plot(
            segment_times_seconds,
            gas_series,
            color="#9467bd",
            linewidth=1.5,
            linestyle="-.",
            label="Throttle",
        )
        ax_push_index.plot(
            segment_times_seconds,
            brake_series,
            color="#8c564b",
            linewidth=1.5,
            linestyle=":",
            label="Brake",
        )
        ax_push_index.set_title("Driver push-to-limit index")
        ax_push_index.set_xlabel("Segment time (s)")
        ax_push_index.set_ylabel("Push index / Pedal (0-1)")
        ax_push_index.set_ylim(0.0, 1.05)
        ax_push_index.grid(True, linestyle="--", alpha=0.3)
        ax_push_index.tick_params(labelbottom=True)

        ax_push_steer = ax_push_index.twinx()
        steer_line, = ax_push_steer.plot(
            segment_times_seconds,
            steer_series,
            color="#17becf",
            linewidth=1.5,
            label="Steer angle",
        )
        ax_push_steer.set_ylabel("Steer angle")

        push_legend_handles = ax_push_index.get_lines() + [steer_line]
        push_legend_labels = [line.get_label() for line in push_legend_handles]
        ax_push_index.legend(push_legend_handles, push_legend_labels, loc="upper right")

        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_bytes = buffer.getvalue()

        saved_path: Optional[Path] = None
        prefix = file_name_prefix or "optimal_segment"
        if output_path is not None:
            saved_path = output_path / f"{prefix}_{idx + 1}.png"
            saved_path.write_bytes(image_bytes)

        if show:
            plt.show()

        plt.close(fig)

        summary: Dict[str, Any] = {}
        if analyze_segment_fn is not None:
            try:
                summary_result = analyze_segment_fn(segment_df, expert_context_features + tire_context_features)
                if hasattr(summary_result, "to_dict"):
                    summary = summary_result.to_dict()
                elif dataclasses.is_dataclass(summary_result):
                    summary = dataclasses.asdict(summary_result)
                else:
                    summary = summary_result
            except Exception as analyze_error:
                summary = {"error": str(analyze_error)}

        payload: Dict[str, Any] = {
            "segment_index": idx,
            "record_count": int(len(segment_df)),
            "summary": summary,
        }

        if return_base64:
            payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")

        if saved_path is not None:
            payload["saved_path"] = str(saved_path)

        visualization_payloads.append(payload)

        buffer.close()

    return visualization_payloads


def visualize_segment_position_coverage(
    histogram_counts: Sequence[float],
    bin_edges: Sequence[float],
    *,
    total_points: Optional[int] = None,
    show: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    return_base64: bool = True,
    file_name_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Render coverage histogram for Graphics_normalized_car_position across segments.

    Args:
        histogram_counts: Sequence of counts for each histogram bin (length N).
        bin_edges: Sequence of bin edge values (length N + 1).
        total_points: Optional total number of samples represented by the histogram.
        show: Display figure interactively when True.
        output_dir: Directory to persist plot image as .png file.
        return_base64: Include base64 PNG payload in the response when True.
        file_name_prefix: Prefix for saved figure filename.

    Returns:
        Dictionary containing metadata for the rendered coverage figure.
    """

    counts = np.asarray(histogram_counts, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)

    if counts.ndim != 1 or edges.ndim != 1:
        raise ValueError("Histogram inputs must be one-dimensional sequences")
    if edges.size != counts.size + 1:
        raise ValueError("bin_edges length must be histogram_counts length + 1")

    effective_total = float(total_points) if total_points is not None else float(counts.sum())
    if effective_total <= 0:
        raise ValueError("Total number of samples must be positive to visualize coverage")

    coverage = counts / effective_total
    bin_widths = np.diff(edges)
    bin_centers = edges[:-1] + (bin_widths / 2.0)
    cumulative_coverage = np.cumsum(coverage)

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "output"
            / "transformer_eval"
            / "figures"
        )

    output_path: Optional[Path] = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(
        bin_centers,
        coverage * 100.0,
        width=bin_widths,
        align="center",
        color="#1f77b4",
        edgecolor="#154a72",
        alpha=0.75,
        label="Bin coverage (%)",
    )

    ax.plot(
        bin_centers,
        cumulative_coverage * 100.0,
        color="#ff7f0e",
        linewidth=2.5,
        label="Cumulative coverage (%)",
    )

    ax.set_title("Coverage of Graphics_normalized_car_position across segments")
    ax.set_xlabel("Graphics_normalized_car_position")
    ax.set_ylabel("Coverage (%)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(coverage.max() * 120.0, 5.0))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    # Annotate top bins for quick inspection when coverage is sparse
    significant_bins = np.argsort(counts)[-3:]
    for idx in significant_bins:
        if counts[idx] <= 0:
            continue
        ax.text(
            bin_centers[idx],
            coverage[idx] * 105.0,
            f"{coverage[idx] * 100.0:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=45,
        )

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image_bytes = buffer.getvalue()

    saved_path: Optional[Path] = None
    prefix = file_name_prefix or "segment_position_coverage"
    if output_path is not None:
        saved_path = output_path / f"{prefix}.png"
        saved_path.write_bytes(image_bytes)

    if show:
        plt.show()

    plt.close(fig)

    payload: Dict[str, Any] = {
        "total_points": int(effective_total),
        "bin_edges": edges.tolist(),
        "histogram_counts": counts.tolist(),
    }

    if return_base64:
        payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")

    if saved_path is not None:
        payload["saved_path"] = str(saved_path)

    buffer.close()

    return payload
