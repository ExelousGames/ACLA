"""Streamlit UI for annotating telemetry prompt datasets.

This tool lets race engineers review telemetry windows, visualise samples,
and attach concise coaching explanations before fine-tuning the local LLM.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _ensure_app_module_on_path() -> None:
    """Add the package root that contains `app/` to sys.path when running standalone."""

    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        candidate = candidate.parent


_ensure_app_module_on_path()

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService
from app.services.local_llm_service import GenerationRequest
from app.services.telemetry_prompt_dataset_builder import DatasetBuildStats
from app.services.llm.prompt_response_example import PromptResponseExample

# Default directory where datasets are written by the ML service
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "models" / "llm_datasets"
ANNOTATION_SUFFIX = ".annotations.jsonl"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _run_async(func, *args, **kwargs):
    """Execute an async function from a synchronous context."""

    try:
        return asyncio.run(func(*args, **kwargs))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@st.cache_resource(show_spinner=False)
def get_ml_service() -> Full_dataset_TelemetryMLService:
    """Return a cached instance of the telemetry ML service."""

    return Full_dataset_TelemetryMLService()


@st.cache_resource(show_spinner=False)
def load_guidance_model(provider: str = "local") -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """Load the active LLM guidance model, if available."""

    service = get_ml_service()
    try:
        model, metadata = _run_async(service.get_llm_guidance_model, provider=provider)
        if model is None:
            error_msg = metadata.get("error") if isinstance(metadata, dict) else "No active model found"
            return None, metadata, error_msg
        return model, metadata, None
    except Exception as error:  # pragma: no cover - guarded for runtime issues
        return None, None, str(error)


def _extract_system_prompt(entry: Any) -> str:
    """Return the system prompt stored with a dataset entry."""

    # Check if it's a PromptResponseExample or has system_prompt attribute
    if hasattr(entry, "system_prompt") and entry.system_prompt:
        return entry.system_prompt

    # Handle dictionary-like access
    if hasattr(entry, "get"):
        raw_record = entry.get("raw", {})
        if not isinstance(raw_record, dict):
            return ""

        metadata = raw_record.get("metadata", {}) or {}
        system_prompt = metadata.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt:
            return system_prompt

        messages = raw_record.get("messages", []) if isinstance(raw_record, dict) else []
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return ""


def _get_config_from_env() -> Dict[str, Any]:
	"""Get configuration from environment variables."""
	return {
		"dataset_path": os.getenv("ANNOTATION_DATASET_PATH", ""),
		"dataset_dir": os.getenv("ANNOTATION_DATASET_DIR", ""),
		"server_address": os.getenv("ANNOTATION_UI_SERVER_ADDRESS", "0.0.0.0"),
		"server_port": int(os.getenv("ANNOTATION_UI_SERVER_PORT", "8501")),
		"browser_address": os.getenv("ANNOTATION_UI_BROWSER_ADDRESS", ""),
		"browser_port": int(os.getenv("ANNOTATION_UI_BROWSER_PORT", "0")) if os.getenv("ANNOTATION_UI_BROWSER_PORT") else 0,
	}
def _resolve_network_config(config: Dict[str, Any]) -> Tuple[str, int, Optional[str], Optional[int], str]:
	"""Resolve the host/port pairing used for Streamlit when running in Docker."""

	server_address = config.get("server_address") or "0.0.0.0"
	server_port = config.get("server_port") or 8501

	browser_address = config.get("browser_address") or ""
	browser_port = config.get("browser_port") or 0

	if not browser_address and Path("/.dockerenv").exists():
		browser_address = os.getenv("ANNOTATION_UI_HOST", "localhost")

	effective_host = browser_address or server_address
	effective_port = browser_port or server_port
	access_url = f"http://{effective_host}:{effective_port}"

	return server_address, server_port, browser_address or None, browser_port, access_url
# ---------------------------------------------------------------------------
# Dataset loading / persistence helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> List[PromptResponseExample]:
    """Load a dataset of prompt-response examples from a JSONL file."""
    if not path_str or not Path(path_str).exists():
        return []
    
    records = []
    with Path(path_str).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(PromptResponseExample.from_record(data))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                st.warning(f"Skipping invalid dataset line: {e}")
    return records


@st.cache_data(show_spinner=False)
def load_annotation_store(path_str: str) -> Dict[str, Dict[str, Any]]:
    """Load existing annotations stored alongside the dataset."""

    annotation_path = Path(path_str)
    store: Dict[str, Dict[str, Any]] = {}

    if not annotation_path.exists():
        return store

    with annotation_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            window_id = payload.get("window_id")
            if window_id:
                store[window_id] = {
                    "window_id": window_id,
                    "coaching_explanation": payload.get("coaching_explanation", ""),
                    "updated_at": payload.get("updated_at"),
                }

    return store


def _build_user_prompt(
    window: Dict[str, Any],
    config: Dict[str, Any],
    segment_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Compose the LLM user prompt for segment-purpose explanations."""

    context_payload = window.get("context", []) or []
    future_payload = window.get("future", []) or []
    context_steps = config.get("context_steps", len(context_payload))
    prediction_steps = config.get("prediction_steps", len(future_payload))

    metadata_payload: Dict[str, Any] = {
        "context_steps": context_steps,
        "future_steps": prediction_steps,
    }
    if segment_metadata:
        metadata_payload.update(segment_metadata)

    return (
        "Task: Provide a concise coaching explanation that captures the purpose of this telemetry segment.\n"
        f"Segment metadata:\n{json.dumps(metadata_payload, indent=2, ensure_ascii=False)}\n\n"
        "Telemetry context (ordered timesteps):\n"
        f"{json.dumps(context_payload, indent=2, ensure_ascii=False)}\n\n"
        "Continuation of the segment (if available):\n"
        f"{json.dumps(future_payload, indent=2, ensure_ascii=False)}\n\n"
        "Respond with a JSON object containing:\n"
        "- `coaching_summary`: 2-3 sentences describing the segment's focus or coaching insight.\n"
        "- Optional `key_focus`: list of short bullet strings highlighting the key observations."
    )


def _save_annotation(
    dataset_path: Path,
    annotation_path: Path,
    window_id: str,
    coaching_explanation: str,
) -> Tuple[int, int]:
    """Persist the updated annotation to the dataset."""

    examples = load_dataset(dataset_path.as_posix())
    if not examples:
        return 0, 0

    annotated_examples = 0
    for example in examples:
        if example.metadata and example.metadata.get("window_id") == window_id:
            example.response = coaching_explanation
            if example.metadata:
                example.metadata["annotation_complete"] = True
            else:
                example.metadata = {"annotation_complete": True, "window_id": window_id}
        
        if example.metadata and example.metadata.get("annotation_complete"):
            annotated_examples += 1

    with dataset_path.open("w", encoding="utf-8") as jsonl_file:
        for example in examples:
            jsonl_file.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")

    # Update annotation store for quick lookup, though dataset is source of truth
    store = load_annotation_store(annotation_path.as_posix())
    store[window_id] = {
        "window_id": window_id,
        "coaching_explanation": coaching_explanation,
        "updated_at": datetime.utcnow().isoformat(),
    }
    with annotation_path.open("w", encoding="utf-8") as jsonl_file:
        for record in store.values():
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(examples), annotated_examples


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _prepare_plot_data(window: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    """Flatten context and future windows for plotting."""

    context_records = window.get("context", [])
    future_records = window.get("future", [])

    if not context_records and not future_records:
        return pd.DataFrame()

    context_df = pd.DataFrame(context_records)
    context_df["phase"] = "context"
    future_df = pd.DataFrame(future_records)
    future_df["phase"] = "future"

    # Create sequential timestamp index
    context_df["timestamp_index"] = range(len(context_df))
    if len(future_df) > 0:
        start_offset = len(context_df)
        future_df["timestamp_index"] = range(start_offset, start_offset + len(future_df))

    combined = pd.concat([context_df, future_df], ignore_index=True, sort=False)
    melted = combined.melt(
        id_vars=["timestamp_index", "phase"],
        value_vars=[feature for feature in feature_names if feature in combined.columns],
        var_name="feature",
        value_name="value",
    )
    return melted


def _extract_features_from_segment(segment: List[Dict[str, Any]]) -> List[str]:
    """Extract all unique feature keys from a segment."""
    
    if not segment or not isinstance(segment, list):
        return []
    
    # Get all keys from the first non-empty dictionary in the segment
    for step in segment:
        if isinstance(step, dict) and step:
            # Return all keys, filtering out common metadata fields
            excluded_fields = {"relative_index", "timestamp", "time", "index"}
            features = [key for key in step.keys() if key not in excluded_fields]
            return sorted(features)
    
    return []


def _render_plot(data: pd.DataFrame) -> None:
    """Render telemetry lines using Plotly."""

    if data.empty:
        st.info("No telemetry data available for plotting.")
        return

    fig = px.line(
        data,
        x="timestamp_index",
        y="value",
        color="feature",
        line_dash="phase" if "phase" in data.columns else None,
        markers=True,
    )
    fig.update_layout(
        height=360,
        legend_orientation="h",
        legend_y=-0.2,
        xaxis_title="Timestamp Index",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_3d_position_comparison(window: Dict[str, Any]) -> None:
    """Render 3D comparison of expert optimal vs player positions."""
    
    context_records = window.get("context", [])
    future_records = window.get("future", [])
    
    if not context_records and not future_records:
        st.info("No telemetry data available for 3D position comparison.")
        return
    
    # Combine context and future records
    all_records = context_records + future_records
    
    # Check if position data exists
    has_expert = any("expert_optimal_player_pos_x" in rec for rec in all_records)
    has_player = any("Graphics_player_pos_x" in rec for rec in all_records)
    
    if not (has_expert or has_player):
        st.info("No position data (expert or player) available for 3D visualization.")
        return
    
    fig = go.Figure()
    
    # Extract expert optimal position
    if has_expert:
        expert_x = [rec.get("expert_optimal_player_pos_x") for rec in all_records if "expert_optimal_player_pos_x" in rec]
        expert_y = [rec.get("expert_optimal_player_pos_y") for rec in all_records if "expert_optimal_player_pos_y" in rec]
        expert_z = [rec.get("expert_optimal_player_pos_z") for rec in all_records if "expert_optimal_player_pos_z" in rec]
        
        if expert_x and expert_y and expert_z:
            # Add trajectory line
            fig.add_trace(go.Scatter3d(
                x=expert_x,
                y=expert_y,
                z=expert_z,
                mode='lines+markers',
                name='Expert Optimal',
                line=dict(color='green', width=4),
                marker=dict(size=4, color='green')
            ))
            
            # Add start point marker
            fig.add_trace(go.Scatter3d(
                x=[expert_x[0]],
                y=[expert_y[0]],
                z=[expert_z[0]],
                mode='markers',
                name='Expert Start',
                marker=dict(size=12, color='lime', symbol='diamond', 
                           line=dict(color='darkgreen', width=2)),
                showlegend=True
            ))
            
            # Add end point marker
            fig.add_trace(go.Scatter3d(
                x=[expert_x[-1]],
                y=[expert_y[-1]],
                z=[expert_z[-1]],
                mode='markers',
                name='Expert End',
                marker=dict(size=12, color='darkgreen', symbol='square',
                           line=dict(color='lime', width=2)),
                showlegend=True
            ))

    # Extract player position
    if has_player:
        player_x = [rec.get("Graphics_player_pos_x") for rec in all_records if "Graphics_player_pos_x" in rec]
        player_y = [rec.get("Graphics_player_pos_y") for rec in all_records if "Graphics_player_pos_y" in rec]
        player_z = [rec.get("Graphics_player_pos_z") for rec in all_records if "Graphics_player_pos_z" in rec]
        
        if player_x and player_y and player_z:
            # Add trajectory line
            fig.add_trace(go.Scatter3d(
                x=player_x,
                y=player_y,
                z=player_z,
                mode='lines+markers',
                name='Player Actual',
                line=dict(color='blue', width=4),
                marker=dict(size=4, color='blue')
            ))
            
            # Add start point marker
            fig.add_trace(go.Scatter3d(
                x=[player_x[0]],
                y=[player_y[0]],
                z=[player_z[0]],
                mode='markers',
                name='Player Start',
                marker=dict(size=12, color='cyan', symbol='diamond',
                           line=dict(color='darkblue', width=2)),
                showlegend=True
            ))
            
            # Add end point marker
            fig.add_trace(go.Scatter3d(
                x=[player_x[-1]],
                y=[player_y[-1]],
                z=[player_z[-1]],
                mode='markers',
                name='Player End',
                marker=dict(size=12, color='darkblue', symbol='square',
                           line=dict(color='cyan', width=2)),
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title="3D Position Comparison: Expert Optimal vs Player Actual",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position",
            aspectmode='data'
        ),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def _generate_explanation_callback(
    ml_service: Full_dataset_TelemetryMLService,
    llm_model: Any,
    selected_entry: Dict[str, Any],
    config: Dict[str, Any],
    coaching_key: str,
    key_focus_key: str,
    raw_output_key: str,
    hf_model_id: Optional[str],
) -> None:
    """Callback to generate explanation and update session state."""
    try:
        system_prompt = _extract_system_prompt(selected_entry)
        metadata = selected_entry.get("metadata", {}) or {}
        user_prompt = _build_user_prompt(
            selected_entry.get("window", {}),
            config,
            metadata.get("segment_metadata"),
        )
        request = GenerationRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=ml_service.llm_config.generation_max_new_tokens,
            temperature=ml_service.llm_config.generation_temperature,
            top_p=ml_service.llm_config.generation_top_p,
            do_sample=ml_service.llm_config.generation_do_sample,
            model_id=hf_model_id,
        )

        raw_output = llm_model.generate(request)
        commentary = ml_service._parse_llm_output(raw_output)
        
        summary_text = commentary.get("coaching_summary") or commentary.get("summary") or raw_output
        st.session_state[coaching_key] = summary_text.strip()
        
        key_focus = commentary.get("key_focus")
        if isinstance(key_focus, list) and key_focus:
            st.session_state[key_focus_key] = key_focus
        else:
            st.session_state.pop(key_focus_key, None)
            
        st.session_state[raw_output_key] = raw_output
        
    except Exception as inference_error:
        # We can't use st.error here easily as it might not render where we want
        # Instead we could set a session state error variable
        print(f"LLM generation failed: {inference_error}")


def main() -> None:
	config = _get_config_from_env()
	server_address, server_port, browser_address, browser_port, access_url = _resolve_network_config(config)

	if os.environ.get("ANNOTATION_UI_URL_PRINTED") != access_url:
		print(f"Telemetry annotation UI will be reachable at {access_url}", flush=True)
		os.environ["ANNOTATION_UI_URL_PRINTED"] = access_url

	st.set_page_config(page_title="Telemetry Prompt Annotation", layout="wide")
	st.title("Telemetry Prompt Annotation")
	st.sidebar.info(f"Open {access_url} from the host browser")
	st.sidebar.caption("Override via ANNOTATION_UI_* environment variables.")

	st.sidebar.markdown("---")
	st.sidebar.subheader("Inference Settings")
	provider_option = st.sidebar.radio(
		"Model Provider",
		options=["Local", "Hugging Face Cloud"],
		index=0,
		help="Select 'Local' to use the locally trained adapter, or 'Hugging Face Cloud' to use the model hosted on HF Hub."
	)
	provider_key = "cloud" if provider_option == "Hugging Face Cloud" else "local"
	
	hf_model_id = None
	if provider_key == "cloud":
		hf_model_id = st.sidebar.text_input(
			"Hugging Face Model ID",
			value="",
			placeholder="username/custom-model-name",
			help="Enter the ID of your fine-tuned model on Hugging Face (e.g., 'username/autotrain-my-model'). If left empty, the base model will be used."
		)

	ml_service = get_ml_service()
	llm_model, llm_metadata, llm_error = load_guidance_model(provider=provider_key)
	llm_available = llm_model is not None

	if llm_available:
		st.sidebar.success("LLM guidance model loaded")
		if llm_metadata and isinstance(llm_metadata, dict):
			adapter_name = llm_metadata.get("backend_metadata", {}).get("adapter_directory")
			if adapter_name:
				st.sidebar.caption(f"Adapter: {adapter_name}")
	else:
		st.sidebar.info("LLM auto-coaching disabled. Train or import a guidance model to enable suggestions.")
		if llm_error:
			st.sidebar.caption(f"LLM load issue: {llm_error}")

	if llm_error:
		st.warning(f"Auto-coaching is currently unavailable: {llm_error}")

	dataset_dir = Path(config.get("dataset_dir") or DEFAULT_DATASET_DIR)
	dataset_dir.mkdir(parents=True, exist_ok=True)

	st.sidebar.markdown("---")
	st.sidebar.subheader("Annotate Existing Dataset")

	# Filter out annotation files to show only source datasets
	all_files = dataset_dir.rglob("*.jsonl")
	dataset_files = sorted(
		[f for f in all_files if not f.name.endswith(ANNOTATION_SUFFIX)],
		key=lambda p: p.stat().st_mtime,
		reverse=True
	)

	if not dataset_files:
		st.warning(
			f"No dataset files found in {dataset_dir}. Run the telemetry training pipeline to generate an annotation dataset."
		)
		return

	dataset_labels = [str(file.relative_to(dataset_dir)) for file in dataset_files]

	# Initialize selection in session state if not present
	if "selected_dataset_label" not in st.session_state:
		default_label = dataset_labels[0]
		if config.get("dataset_path"):
			target_name = Path(config["dataset_path"]).name
			for label in dataset_labels:
				if label.endswith(target_name):
					default_label = label
					break
		st.session_state["selected_dataset_label"] = default_label

	# Ensure the selected label is still valid
	if st.session_state.get("selected_dataset_label") not in dataset_labels:
		st.session_state["selected_dataset_label"] = dataset_labels[0]

	selected_label = st.sidebar.selectbox(
		"Dataset file", 
		dataset_labels, 
		key="selected_dataset_label"
	)
	dataset_path = dataset_files[dataset_labels.index(selected_label)]
	annotation_path = dataset_path.with_suffix(dataset_path.suffix + ANNOTATION_SUFFIX)

	entries = load_dataset(dataset_path.as_posix())
	if not entries:
		st.warning("Dataset is empty or could not be parsed.")
		return

	store = load_annotation_store(annotation_path.as_posix())

	total_examples = len(entries)
	annotated_examples = sum(1 for entry in entries if entry.get("annotation_complete"))

	active_dataset_key = "_active_annotation_dataset"
	dataset_identifier = dataset_path.as_posix()
	if st.session_state.get(active_dataset_key) != dataset_identifier:
		st.session_state[active_dataset_key] = dataset_identifier

	st.sidebar.metric(
		"Annotated",
		f"{annotated_examples}/{total_examples}",
		"{:.0%}".format(annotated_examples / total_examples) if total_examples else "0%",
	)
	show_pending_only = st.sidebar.checkbox("Show only pending annotations", value=False)

	entries_to_view = list(entries)
	if show_pending_only:
		pending_entries = [entry for entry in entries_to_view if not entry.get("annotation_complete")]
		if pending_entries:
			entries_to_view = pending_entries
		else:
			st.success("All windows are annotated!")

	summary_df = pd.DataFrame(
		[
			{
				"Window ID": entry.get("window_id", "Unknown"),
				"Annotated": "Yes" if entry.get("annotation_complete") else "No",
				"Existing Explanation": (entry.get("coaching_explanation", "")[:50] + "…") if entry.get("coaching_explanation") else "",
			}
			for entry in entries_to_view
		]
	)
	st.dataframe(summary_df, use_container_width=True, height=260)

	window_ids = [entry.get("window_id") for entry in entries_to_view if entry.get("window_id")]
	if not window_ids:
		st.error("No windows available for annotation.")
		return

	selected_window_id = st.selectbox("Select window", window_ids)
	selected_entry = next((entry for entry in entries if entry.get("window_id") == selected_window_id), None)
	if not selected_entry:
		st.error("Unable to locate the selected window in the dataset.")
		return

	config = selected_entry.get("config", {})
	
	# Try to get features from config first, then extract from segment data
	feature_options = config.get("telemetry_features", [])
	if not feature_options:
		# Extract features directly from the segment
		segment = selected_entry.get("segment", [])
		if not segment:
			# Try to get from window context
			window = selected_entry.get("window", {})
			segment = window.get("context", []) if isinstance(window, dict) else []
		
		feature_options = _extract_features_from_segment(segment)
	
	# Set default features to Physics_gas, Physics_brake, and Physics_steer_angle
	default_features = ["Physics_gas", "Physics_brake", "Physics_steer_angle"]
	# Only use defaults that exist in feature_options
	default_selection = [f for f in default_features if f in feature_options]
	# If none of the preferred defaults exist, fall back to first 5 features
	if not default_selection:
		default_selection = feature_options[: min(5, len(feature_options))]
	
	selected_features = st.multiselect(
		"Telemetry features to display",
		feature_options,
		default=default_selection,
	)

	if selected_features:
		plot_data = _prepare_plot_data(selected_entry.get("window", {}), selected_features)
		_render_plot(plot_data)
	else:
		st.info("Select at least one telemetry feature to visualise.")
	
	# Add 3D position comparison visualization
	st.subheader("3D Position Comparison")
	window_data = selected_entry.get("window", {})
	_render_3d_position_comparison(window_data)

	existing_annotation = store.get(selected_window_id, {})
	coaching_note_default = existing_annotation.get("coaching_explanation") or selected_entry.get("coaching_explanation", "")

	st.subheader("Annotation")

	active_window_key = "_active_annotation_window"
	coaching_key = f"coaching_note_{selected_window_id}"
	key_focus_key = f"{coaching_key}_key_focus"
	raw_output_key = f"{coaching_key}_raw_output"

	if st.session_state.get(active_window_key) != selected_window_id:
		st.session_state[active_window_key] = selected_window_id
		st.session_state[coaching_key] = coaching_note_default
		st.session_state.pop(key_focus_key, None)
		st.session_state.pop(raw_output_key, None)
	else:
		st.session_state.setdefault(coaching_key, coaching_note_default)

	coaching_explanation = st.text_area("Coaching explanation", key=coaching_key, height=160)

	actions_col1, actions_col2, stats_col = st.columns([1, 1, 1])

	if llm_error and st.session_state.get(raw_output_key):
		st.session_state.pop(raw_output_key, None)
		st.session_state.pop(key_focus_key, None)

	with actions_col1:
		generate_disabled = bool(llm_error) or not llm_available
		generate_help = None if llm_available and not llm_error else (llm_error or "Train or import a guidance model to enable auto-suggestions.")
		
		st.button(
			"Generate explanation",
			type="secondary",
			disabled=generate_disabled,
			help=generate_help,
			use_container_width=True,
			on_click=_generate_explanation_callback,
			args=(
				ml_service,
				llm_model,
				selected_entry,
				config,
				coaching_key,
				key_focus_key,
				raw_output_key,
				hf_model_id,
			),
		)

	with actions_col2:
		# Track if save button was just clicked to prevent duplicate saves during rerun
		save_clicked_key = f"_save_clicked_{selected_window_id}"
		
		if st.button("Save annotation", type="primary", use_container_width=True, key=f"save_btn_{selected_window_id}"):
			# Only save if we haven't just saved in the previous frame
			if not st.session_state.get(save_clicked_key, False):
				st.session_state[save_clicked_key] = True
				total, annotated = _save_annotation(
					dataset_path=dataset_path,
					annotation_path=annotation_path,
					window_id=selected_window_id,
					coaching_explanation=coaching_explanation.strip(),
				)
				# Clear caches before rerun
				load_dataset.clear()
				load_annotation_store.clear()
				st.success(f"Annotation saved for {selected_window_id}")
				st.rerun()
		else:
			# Reset the flag when button is not clicked
			st.session_state[save_clicked_key] = False

	with stats_col:
		st.metric(
			"Annotation progress",
			f"{annotated_examples}/{total_examples}",
			"{:.0%}".format(annotated_examples / total_examples) if total_examples else "0%",
		)

	# Add finish session button
	st.markdown("---")
	finish_col1, finish_col2 = st.columns([3, 1])
	with finish_col1:
		st.info("⚠️ When done annotating, click 'Finish Session' to close the app and continue training.")
	with finish_col2:
		if st.button("✅ Finish Session", type="primary", use_container_width=True):
			st.success("Session complete! Closing annotation app...")
			st.balloons()
			import time
			time.sleep(1)
			# Exit the Streamlit process cleanly
			os._exit(0)

	suggested_focus = st.session_state.get(key_focus_key)
	if suggested_focus:
		st.markdown("**Suggested focus areas**")
		for item in suggested_focus:
			st.markdown(f"- {item}")

	raw_output = st.session_state.get(raw_output_key)
	if raw_output:
		with st.expander("Raw LLM output", expanded=False):
			st.code(raw_output, language="json")

	st.caption(
		"Annotations are written back to the dataset file and a companion annotations log."
		" Training can proceed once the necessary windows have notes and explanations."
	)


def run_annotation_app(dataset_path: str, dataset_dir: str) -> None:
	"""Programmatic entry point for launching the annotation app.
	
	Args:
		dataset_path: Path to the specific dataset file to annotate
		dataset_dir: Directory containing dataset files
	"""
	import streamlit.web.bootstrap as bootstrap
	
	# Set environment variables for the app
	os.environ["ANNOTATION_DATASET_PATH"] = str(dataset_path)
	os.environ["ANNOTATION_DATASET_DIR"] = str(dataset_dir)
	os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
	
	# Get the path to this script
	script_path = Path(__file__).resolve()
	
	print(f"[INFO] Starting Streamlit annotation app for {dataset_path}")
	
	# Run the Streamlit app programmatically
	bootstrap.run(
		str(script_path),
		"",  # command_line args
		[],  # args
		{},  # flag_options
	)


if __name__ == "__main__":
	# Check if we have dataset path in environment
	dataset_path = os.environ.get("ANNOTATION_DATASET_PATH")
	dataset_dir = os.environ.get("ANNOTATION_DATASET_DIR")
	
	if dataset_path and dataset_dir:
		# Called with environment variables set - set config and run main
		# Don't use bootstrap.run here as it would create nested Streamlit instances
		main()
	else:
		# Called directly, just run main
		main()