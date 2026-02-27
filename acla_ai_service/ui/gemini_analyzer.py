import os
import time
import io
import base64
import threading
import json
import sys
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import torch
torch.classes.__path__ = []
import matplotlib
# Use Agg backend for non-interactive (headless) plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from app.models.segment_models import MAIN_LABEL_GUIDELINES, LABEL_NAME_TO_ID, LABEL_IMAGE_MAP
except ImportError:
    # Fallback or mock if app module is not in path for standalone testing
    MAIN_LABEL_GUIDELINES = {}
    LABEL_NAME_TO_ID = {}
    LABEL_IMAGE_MAP = {}
from google import genai
from PIL import Image
import streamlit as st

# Predefined Graph List with Descriptions for Gemini Context
GRAPH_DEFINITIONS = [
    {
        "id": "throttle",
        "title": "Throttle Application: Expert vs Player",
        "columns": ["expert_optimal_throttle", "Physics_gas"],
        "description": "Compares the driver's throttle application (Physics_gas) against the optimal expert line. Focus on timing of throttle application and lift-off."
    },
    {
        "id": "brake",
        "title": "Brake Application: Expert vs Player",
        "columns": ["expert_optimal_brake", "Physics_brake"],
        "description": "Compares the driver's braking (Physics_brake) against the optimal expert line. Look for differences in braking points and brake pressure modulation."
    },
    {
        "id": "time_delta",
        "title": "Time Difference to Expert",
        "columns": ["expert_time_difference"],
        "description": "Instantaneous time delta compared to the expert. Positive values mean the driver is slower/behind the expert at that specific point."
    },
    {
        "id": "speed_delta",
        "title": "Speed Difference (Expert - Player)",
        "columns": ["speed_difference"],
        "description": "Difference in speed between expert and player. Positive values indicate the expert is faster."
    },
    {
        "id": "speed",
        "title": "Speed Trace: Expert vs Player",
        "columns": ["expert_optimal_speed", "Physics_speed_kmh"],
        "description": "Absolute speed comparison between the driver and expert."
    },
    {
        "id": "push_limit",
        "title": "Driver Push/Limit",
        "columns": ["driver_push_to_limit"],
        "description": "Metric indicating how close the driver is pushing to the vehicle limit. Over 1 means the driver is pushing beyond the estimated limit, which could indicate tire slip. Less than 1 means the driver is not fully utilizing the potential grip."
    },
    {
        "id": "trajectory_context",
        "title": "Track Map Overview",
        "columns": [], 
        "description": "Shows where the segment (green) is located on the full track (grey). Combine with track reference map to identify specific corners or sections."
    },
    {
        "id": "trajectory_detailed",
        "title": "Detailed Trajectory",
        "columns": [],
        "description": "Close-up of path, showing corner apex and minimum speed points. Green is player, Blue is expert reference. Look for differences in line choice (early/late apex)."
    },
    {
        "id": "track_map",
        "title": "Track Map",
        "columns": [],
        "description": "track reference map image for context. Compare the segment's trajectory to the track layout to identify specific corners or sections."
    }
]

class GeminiAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not api_key:
            raise ValueError("API Key is required")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-3-pro-preview'  # or the latest available Gemini model

    @staticmethod
    def get_api_key() -> Optional[str]:
        """
        Retrieves the Gemini API key from environment variables or prompts the user via Streamlit input.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_api_key_input")
        return api_key

    @staticmethod
    def get_max_curvature_point(df_in, x_col, y_col, z_col=None, speed_col=None, label_type="Player"):
            if df_in.empty or len(df_in) <= 5:
                # Need enough points
                return None
            try:
                xs = df_in[x_col].values
                ys = df_in[y_col].values
                
                dx = np.gradient(xs)
                dy = np.gradient(ys)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                
                # Curvature k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
                numerator = np.abs(dx * ddy - dy * ddx)
                denominator = np.power(dx**2 + dy**2, 1.5)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    curvature = np.where(denominator > 1e-6, numerator / denominator, 0)
                
                curvature = np.nan_to_num(curvature)

                # Ignore low speed noise
                if speed_col and speed_col in df_in.columns:
                    s_vals = df_in[speed_col].values
                    curvature[s_vals < 10] = 0.0

                max_k = np.max(curvature)
                
                # If we detect a corner (significant curvature)
                if max_k > 0.002: # Threshold for corner detection
                    # Find point with minimum speed
                    if speed_col and speed_col in df_in.columns:
                        min_speed_idx_local = np.argmin(df_in[speed_col].values)
                        target_idx = df_in.index[min_speed_idx_local]
                        target_row = df_in.loc[target_idx]
                        marker_label = "Corner Apex (Min Speed)"
                    else:
                        # Fallback to max curvature if no speed
                        max_k_idx_local = np.argmax(curvature)
                        target_idx = df_in.index[max_k_idx_local]
                        target_row = df_in.loc[target_idx]
                        marker_label = "Max Curvature"
                    
                    p_geo = {
                        "x": target_row[x_col],
                        "y": target_row[y_col],
                        "Type": label_type,
                        "ID": f"{label_type} {marker_label}",
                        "Marker": marker_label,
                        "Speed": target_row[speed_col] if speed_col and speed_col in df_in.columns else None
                    }
                    if z_col and z_col in df_in.columns:
                        p_geo["z"] = target_row[z_col]
                    return p_geo
            except Exception:
                pass
            return None

    def _plot_to_image(self, fig) -> Image.Image:
        """Converts a matplotlib figure to a PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        return img

    def create_feature_plot(self, df: pd.DataFrame, columns: List[str], title: str) -> Optional[Image.Image]:
        """Creates a line plot for specific features."""
        if df.empty or not columns:
            return None
        
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        for col in valid_cols:
            ax.plot(df.index, df[col], label=col)
        
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.legend()
        ax.grid(True)
        
        return self._plot_to_image(fig)

    def create_context_trajectory_plot(self, df: pd.DataFrame, track_config: Dict[str, str], context_df: Optional[pd.DataFrame] = None, context_padding: int = 500) -> Optional[Image.Image]:
        """
        Creates a 'Big Picture' trajectory plot showing the segment's location on the track.
        """
        if df.empty or context_df is None or context_df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(8, 8))

        # 0. Context (Background Track)
        if "player_x" in track_config and "player_y" in track_config:
            cpx = track_config["player_x"]
            cpy = track_config["player_y"]
            
            plot_context_df = context_df
            
            if cpx in plot_context_df.columns and cpy in plot_context_df.columns:
                 ax.plot(plot_context_df[cpx], plot_context_df[cpy], label="Context", color="lightgray", linewidth=1.5, linestyle="-")
            
            # Apply Zoom/Padding logic
            # To show the WHOLE track, we use the limits from the context_df (full track), not the local segment df.
            # However, if context_df is just a small slice around the segment (which 'context_padding' implied previously),
            # this logic might differ based on user intent. But "Track Location Context" usually means where on the full map.
            if cpx in plot_context_df.columns and cpy in plot_context_df.columns:
                 min_x, max_x = plot_context_df[cpx].min(), plot_context_df[cpx].max()
                 min_y, max_y = plot_context_df[cpy].min(), plot_context_df[cpy].max()
                    
                 # Determine a padding relevant to the track size (e.g. 5%)
                 width = max_x - min_x
                 height = max_y - min_y
                 pad_x = width * 0.05
                 pad_y = height * 0.05
                 
                 ax.set_xlim(min_x - pad_x, max_x + pad_x)
                 ax.set_ylim(min_y - pad_y, max_y + pad_y)
        
        # Player Segment
        if "player_x" in track_config and "player_y" in track_config:
            px = track_config["player_x"]
            py = track_config["player_y"]
            if px in df.columns and py in df.columns:
                ax.plot(df[px], df[py], label="Segment", color="green", linewidth=3)
                # Mark start
                ax.scatter(df[px].iloc[0], df[py].iloc[0], marker='x', color='black', label='Start')
                
                # Mark end with dot
                ax.scatter(df[px].iloc[-1], df[py].iloc[-1], marker='o', color='black', label='End')

        ax.set_title("Track Location Context")
        ax.invert_yaxis()
        ax.legend()
        ax.set_aspect('equal', 'box')
        ax.axis('off') # Cleaner look for map
        
        return self._plot_to_image(fig)

    def create_detailed_trajectory_plot(self, df: pd.DataFrame, track_config: Dict[str, str]) -> Optional[Image.Image]:
        """
        Creates a detailed trajectory plot focusing on the segment and apex/min-speed points.
        """
        if df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(8, 8))

        # Player
        if "player_x" in track_config and "player_y" in track_config:
            px = track_config["player_x"]
            py = track_config["player_y"]
            if px in df.columns and py in df.columns:
                ax.plot(df[px], df[py], label="Player", color="green", linewidth=2)
                # Mark start
                ax.scatter(df[px].iloc[0], df[py].iloc[0], marker='x', color='green', label='Start')
                
                # Mark end with dot
                ax.scatter(df[px].iloc[-1], df[py].iloc[-1], marker='o', color='green', label='End')

                # Calculate Apex/Curve point (Min Speed)
                apex_data = self.get_max_curvature_point(
                    df, 
                    px, 
                    py, 
                    speed_col="Physics_speed_kmh", 
                    label_type="Player"
                )
                if apex_data:
                    ax.scatter(apex_data["x"], apex_data["y"], marker='*', s=200, color='purple', label='Player Apex', zorder=5)
                    # Annotate speed if available
                    if "Speed" in apex_data and apex_data["Speed"] is not None:
                        ax.annotate(f"Min: {apex_data['Speed']:.1f}", (apex_data["x"], apex_data["y"]), xytext=(10, 10), textcoords='offset points', fontsize=10, color='purple', fontweight='bold')

        # Expert (Reference)
        if "expert_x" in track_config and "expert_y" in track_config:
            ex = track_config["expert_x"]
            ey = track_config["expert_y"]
            if ex in df.columns and ey in df.columns:
                 ax.plot(df[ex], df[ey], label="Expert Line", color="blue", linestyle="--", alpha=0.6)

                 # Calculate Expert Apex (Min Speed)
                 expert_apex_data = self.get_max_curvature_point(
                    df, 
                    ex, 
                    ey, 
                    speed_col="expert_optimal_speed", 
                    label_type="Expert"
                 )
                 if expert_apex_data:
                    ax.scatter(expert_apex_data["x"], expert_apex_data["y"], marker='*', s=200, color='orange', label='Expert Apex', zorder=5)
                    # Annotate expert speed

                    if "Speed" in expert_apex_data and expert_apex_data["Speed"] is not None:
                         ax.annotate(f"Exp Min: {expert_apex_data['Speed']:.1f}", (expert_apex_data["x"], expert_apex_data["y"]), xytext=(10, -20), textcoords='offset points', fontsize=9, color='orange', fontweight='bold')

        # Opponents (Optional, kept for completeness but low alpha)
        for i in range(1, 6):
            ox = f"Opponent_{i}_pos_x"
            oy = f"Opponent_{i}_pos_y"
            if ox in df.columns and oy in df.columns:
                 if df[ox].abs().max() > 0.1 or df[oy].abs().max() > 0.1:
                    ax.plot(df[ox], df[oy], label=f"Opponent {i}", color="red", alpha=0.3)

        ax.set_title("Detailed Segment Trajectory")
        ax.invert_yaxis()
        ax.legend()
        ax.set_aspect('equal', 'box')
        ax.grid(True)

        return self._plot_to_image(fig)

    def _load_label_map_images(self, current_labels: List[str], graph_definitions: List[Dict[str, Any]] = None, track_config: Dict[str, str] = {}, start_index: int = 1) -> tuple[List[Image.Image], List[str]]:
        """
        Loads reference images based on LABEL_IMAGE_MAP.
        - Uses track_config['track_name'] to load track reference map if available
        - Uses current_labels to load any label-specific images
        Loads images from ui/source/ directory.
        Returns (list_of_images, list_of_descriptions).
        """
        if not LABEL_IMAGE_MAP:
            return ([], [])

        graphs_to_process = graph_definitions if graph_definitions else GRAPH_DEFINITIONS
        
        # Helper to find description from GRAPH_DEFINITIONS
        def get_desc(graph_id):
            for g in graphs_to_process:
                if g.get("id") == graph_id:
                    return g.get("description", "")
            return ""

        images = []
        descriptions = []
        
        # Map filename -> trigger info (for tracking what triggered the image load)
        filename_to_info = {}

        # 1. Check track_name from track_config (direct comparison with LABEL_IMAGE_MAP keys)
        if track_config and "track_name" in track_config:
            track_name = track_config["track_name"]
            if track_name in LABEL_IMAGE_MAP:
                filename = LABEL_IMAGE_MAP[track_name]
                if filename not in filename_to_info:
                    filename_to_info[filename] = {"is_track_map": True, "trigger": track_name}
        
        # 2. Check current_labels for any label-specific images
        if current_labels:
            for label in current_labels:
                filename = None
                # Direct lookup in LABEL_IMAGE_MAP
                if label in LABEL_IMAGE_MAP:
                    filename = LABEL_IMAGE_MAP[label]
                # Check if label name maps to ID which is in LABEL_IMAGE_MAP
                elif label in LABEL_NAME_TO_ID:
                    lid = LABEL_NAME_TO_ID[label]
                    if lid in LABEL_IMAGE_MAP:
                        filename = LABEL_IMAGE_MAP[lid]
                
                if filename and filename not in filename_to_info:
                    filename_to_info[filename] = {"is_track_map": False, "trigger": label}
        
        # 3. Load images from filesystem
        possible_paths = [
            os.path.join(os.getcwd(), "acla_ai_service/ui/source"),
            os.path.join(os.getcwd(), "ui/source"),
            os.path.join(os.path.dirname(__file__), "source"),
            "ui/source",
            "source"
        ]
        
        for filename, info in filename_to_info.items():
            loaded = False
            for base_path in possible_paths:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    try:
                        temp_img = Image.open(full_path).convert('RGB')
                        
                        # Determine description based on image type
                        if info["is_track_map"]:
                            # Track reference image - use track_map description
                            description_to_use = get_desc("track_map")
                        else:
                            # Label-specific image - try to find description based on trigger
                            description_to_use = get_desc(info["trigger"])
                        
                        if description_to_use:
                            images.append(temp_img)
                            current_img_num = start_index + len(images) - 1
                            descriptions.append(f"Image {current_img_num}: {description_to_use}")
                            loaded = True
                            break
                    except Exception as e:
                        print(f"Error loading image {full_path}: {e}")
                        break
            
            if not loaded:
                print(f"Warning: Could not load reference image '{filename}' for {info['trigger']}")

        return images, descriptions

    def _prepare_visual_content(self, df: pd.DataFrame, graph_definitions: List[Dict[str, Any]] = None, track_config: Dict[str, str] = {}, context_df: pd.DataFrame = None, context_padding: int = 2000) -> tuple[List[Image.Image], List[str]]:
        """Helper to generate images and descriptions shared between analysis methods."""
        images = []
        plot_descriptions = []
        
        graphs_to_process = graph_definitions if graph_definitions else GRAPH_DEFINITIONS

        for graph_def in graphs_to_process:
            cols = graph_def.get("columns", [])
            title = graph_def.get("title", f"Graph {graph_def.get('id')}")
            description = graph_def.get("description", "")
            
            valid_cols = [c for c in cols if c in df.columns]

            if valid_cols:
                img = self.create_feature_plot(df, valid_cols, title)
                if img and description:
                    images.append(img)
                    # Use strictly the description from GRAPH_DEFINITIONS
                    plot_descriptions.append(f"Image {len(images)}: {description}")

        # Trajectory
        if "player_x" not in track_config and "Graphics_player_pos_x" in df.columns:
             tc = {
                 "player_x": "Graphics_player_pos_x",
                 "player_y": "Graphics_player_pos_y"
             }
             if "expert_optimal_player_pos_x" in df.columns:
                 tc["expert_x"] = "expert_optimal_player_pos_x"
                 tc["expert_y"] = "expert_optimal_player_pos_y"
             track_config = tc

        if "player_x" in track_config:
            # Helper to find description
            def get_desc(graph_id):
                for g in graphs_to_process:
                    if g.get("id") == graph_id:
                        return g.get("description", "")
                return ""

            # 1. Big Picture Context
            ctx_description = get_desc("trajectory_context")
            if ctx_description and context_df is not None and not context_df.empty:
                ctx_img = self.create_context_trajectory_plot(df, track_config, context_df=context_df, context_padding=context_padding)
                if ctx_img:
                    images.append(ctx_img)
                    plot_descriptions.append(f"Image {len(images)}: {ctx_description}")

            # 2. Detailed Trajectory
            traj_description = get_desc("trajectory_detailed")
            if traj_description:
                traj_img = self.create_detailed_trajectory_plot(df, track_config)
                if traj_img:
                    images.append(traj_img)
                    plot_descriptions.append(f"Image {len(images)}: {traj_description}")
        
        return images, plot_descriptions

    def analyze_segment_json(self, 
                        df: pd.DataFrame, 
                        graph_definitions: List[Dict[str, Any]] = None, 
                        track_config: Dict[str, str] = {},
                        current_labels: List[str] = None,
                        available_sub_labels_context: List[str] = None,
                        context_df: pd.DataFrame = None,
                        context_padding: int = 200) -> Dict[str, Any]:
        """
        Analyzes the segment and returns a JSON structured response.
        Useful for batch processing / auto-annotation.
        """
        images, plot_descriptions = self._prepare_visual_content(df, graph_definitions, track_config, context_df, context_padding=context_padding)

        # Add Map Images if relevant labels are present
        map_imgs, map_descs = self._load_label_map_images(current_labels, graph_definitions, track_config=track_config, start_index=len(images) + 1)
        if map_imgs:
            images.extend(map_imgs)
            plot_descriptions.extend(map_descs)

        if not images:
            return {"error": "No valid graphs generated"}

        # Construct Prompt
        labels_context = f"Current identified labels: {', '.join(current_labels)}" if current_labels else "No labels identified yet."
        
        guidelines_text = ""
        if current_labels:
            guidelines = []
            for label in current_labels:
                if label in MAIN_LABEL_GUIDELINES:
                    guidelines.append(f"- Label '{label}': {MAIN_LABEL_GUIDELINES[label]}")
                elif label in LABEL_NAME_TO_ID:
                    lid = LABEL_NAME_TO_ID[label]
                    if lid in MAIN_LABEL_GUIDELINES:
                        guidelines.append(f"- Label '{label}' ({lid}): {MAIN_LABEL_GUIDELINES[lid]}")
            if guidelines:
                guidelines_text = "Main Label Guidelines:\n" + "\n".join(guidelines)
        
        context_block = ""
        if available_sub_labels_context:
            content = "\n".join(available_sub_labels_context) if isinstance(available_sub_labels_context, list) else str(available_sub_labels_context)
            context_block = f"Available specific sub-labels:\n{content}"

        descriptions_text = "\n".join(plot_descriptions)

        prompt_text = f"""
Analyze the provided telemetry data graphs for a racing simulation segment.
{descriptions_text}

{labels_context}
{guidelines_text}

{context_block}

Task:
1. Examine the telemetry graphs and trajectory map.
2. Identify driving maneuvers, incidents, or patterns.
3. Recommend specific labels/sub-labels from the provided list.
4. Return the result STRICTLY as a JSON object.

JSON Schema:
{{
    "analysis_summary": "Short text summary of the analysis.",
    "suggested_labels": [
        {{
            "label": "Name of the label (e.g., 'Corner Entry Overspeed')",
            "confidence": 0.0 to 1.0,
            "reasoning": "Why this label applies based on the graph data."
        }}
    ],
    "primary_issue_detected": true/false
}}
"""
        
        full_content = [prompt_text] + images
        
        try:
            # Enforce JSON if model supports it, or request it in prompt
            # Set a longer timeout for the generate_content call if supported by the client library, 
            # otherwise relies on default.
            
            # --- Footprint Logging ---
            import sys
            thread_info = f"Thread: {threading.current_thread().name} (ID: {threading.get_ident()})"
            approx_image_size = sum([sys.getsizeof(img.tobytes()) for img in images]) if images else 0
            prompt_size = len(prompt_text.encode('utf-8'))
            print(f"DEBUG: [{thread_info}] Gemini API Request Footprint: ~{prompt_size/1024:.2f}KB text, ~{approx_image_size/1024:.2f}KB images. ({len(images)} images)")
            start_t = time.time()
            # -------------------------

            print(f"DEBUG: [{thread_info}] Calling Gemini API for segment analysis (JSON mode). Model: {self.model_name}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_content,
                config={'response_mime_type': 'application/json'}
            )
            
            # --- Response Footprint ---
            duration = time.time() - start_t
            resp_size = len(response.text.encode('utf-8')) if response.text else 0
            print(f"DEBUG: Gemini API Response Footprint: ~{resp_size/1024:.2f}KB received in {duration:.2f}s.")
            # --------------------------

            print("DEBUG: Gemini API returned successfully (JSON mode).")
            return {
                "response_json": response.text,
                "images": images,
                "prompt": prompt_text,
                "metrics": {
                   "request_text_kb": prompt_size/1024,
                   "request_imgs_kb": approx_image_size/1024,
                   "response_kb": resp_size/1024,
                   "duration_sec": duration
                }
            }
        except Exception as e:
             # Fallback if JSON mode isn't supported by this client version yet
            print(f"DEBUG: JSON mode failed/error: {e}. Retrying with default...")
            try:
                print(f"DEBUG: Calling Gemini API (Text mode).")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_content
                )
                print("DEBUG: Gemini API returned (Text mode).")
                return {
                    "response_json": response.text,
                    "images": images,
                    "prompt": prompt_text
                }
            except Exception as e2:
                print(f"DEBUG: Gemini API failed completely: {e2}")
                return {
                    "error": str(e2),
                    "images": images,
                    "prompt": prompt_text
                }

    def analyze_segment(self, 
                        df: pd.DataFrame, 
                        graph_definitions: List[Dict[str, Any]] = None, 
                        track_config: Dict[str, str] = {},
                        current_labels: List[str] = None,
                        available_sub_labels_context: List[str] = None,
                        context_df: pd.DataFrame = None,
                        context_padding: int = 200) -> Dict[str, Any]:
        """
        Analyzes the segment and returns a dictionary with:
        - 'response': The text response from Gemini
        - 'images': List of PIL Images generated and sent
        - 'prompt': The text prompt sent
        """
        images, plot_descriptions = self._prepare_visual_content(df, graph_definitions, track_config, context_df, context_padding=context_padding)

        # Add Map Images if relevant labels are present
        map_imgs, map_descs = self._load_label_map_images(current_labels, graph_definitions, track_config=track_config, start_index=len(images) + 1)
        if map_imgs:
            images.extend(map_imgs)
            plot_descriptions.extend(map_descs)

        if not images:
            return {
                "response": "No valid graphs could be generated from the data.",
                "images": [],
                "prompt": "N/A"
            }

        # Construct the detailed prompt
        labels_context = f"Current identified labels: {', '.join(current_labels)}" if current_labels else "No labels identified yet."
        
        guidelines_text = ""
        if current_labels:
            guidelines = []
            for label in current_labels:
                # Try direct key
                if label in MAIN_LABEL_GUIDELINES:
                    guidelines.append(f"- Label '{label}': {MAIN_LABEL_GUIDELINES[label]}")
                # Try finding ID from name
                elif label in LABEL_NAME_TO_ID:
                    lid = LABEL_NAME_TO_ID[label]
                    if lid in MAIN_LABEL_GUIDELINES:
                        guidelines.append(f"- Label '{label}' ({lid}): {MAIN_LABEL_GUIDELINES[lid]}")
            
            if guidelines:
                guidelines_text = "Main Label Guidelines:\n" + "\n".join(guidelines)
        context_block = ""
        if available_sub_labels_context:
            # Assume available_sub_labels_context is a list of strings or single string
            content = "\n".join(available_sub_labels_context) if isinstance(available_sub_labels_context, list) else str(available_sub_labels_context)
            context_block = f"""Available specific sub-labels based on current selection:\n{content}"""

        prompt_intro = "Analyze the following telemetry data graphs for a racing simulation segment."
        descriptions_text = "\n".join(plot_descriptions)

        prompt_text = f"""
{prompt_intro}
{descriptions_text}

{labels_context}
{guidelines_text}

{context_block}

Task:
1. Examine the provided telemetry graphs and trajectory map.
2. Identify any specific driving maneuvers, incidents, or patterns.
3. Explain WHY you suggest these labels based on the visual evidence in the graphs.
4. Based on the 'Main Label Guidelines' provided above, identify the most appropriate sub-labels from the 'Available specific sub-labels' list.
   Prioritise selecting from the listed sub-labels if they match the data and the guidelines.
5. Provide your output as a concise list of suggestions with brief reasoning.
"""
        
        # Prepare content for Gemini (interleaved text and images as supported by library, or list)
        # The python library supports list of [text, img1, img2...]
        full_content = [prompt_text] + images
        
        try:
            thread_info = f"Thread: {threading.current_thread().name} (ID: {threading.get_ident()})"
            print(f"DEBUG: [{thread_info}] Calling Gemini API for segment analysis (Text mode). Model: {self.model_name}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_content
            )
            print(f"DEBUG: [{thread_info}] Gemini API returned (Text mode).")
            return {
                "response": response.text,
                "images": images,
                "prompt": prompt_text
            }
        except Exception as e:
            return {
                "response": f"Error communicating with Gemini AI: {str(e)}",
                "images": images,
                "prompt": prompt_text
            }

    # ========================================
    # BATCH API FILE JOB METHODS
    # ========================================

    def _image_to_base64(self, img: Image.Image, format: str = "PNG") -> str:
        """Converts a PIL Image to base64 string."""
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def prepare_batch_request(
        self, 
        df: pd.DataFrame, 
        segment_index: int,
        graph_definitions: List[Dict[str, Any]] = None, 
        track_config: Dict[str, str] = {},
        current_labels: List[str] = None,
        available_sub_labels_context: List[str] = None,
        context_df: pd.DataFrame = None,
        include_graphs: bool = True,
        context_padding: int = 2000
    ) -> Optional[Dict[str, Any]]:
        """
        Prepares a single batch request for file-based batch job.
        Returns a dict with 'key' and 'request' fields for JSONL format.
        
        Args:
            segment_index: Index of the segment (used as key identifier)
            include_graphs: If False, skips graph generation and sends text-only request.
            
        Returns:
            Dict with format: {"key": "segment-{idx}", "request": {...}}
        """
        images = []
        plot_descriptions = []
        
        if include_graphs:
            images, plot_descriptions = self._prepare_visual_content(df, graph_definitions, track_config, context_df, context_padding=context_padding)

            # Add Map Images if relevant labels are present
            map_imgs, map_descs = self._load_label_map_images(current_labels, graph_definitions, track_config=track_config, start_index=len(images) + 1)
            if map_imgs:
                images.extend(map_imgs)
                plot_descriptions.extend(map_descs)

        # Construct Prompt (same as analyze_segment_json)
        labels_context = f"Current identified labels: {', '.join(current_labels)}" if current_labels else "No labels identified yet."
        
        guidelines_text = ""
        if current_labels:
            guidelines = []
            for label in current_labels:
                if label in MAIN_LABEL_GUIDELINES:
                    guidelines.append(f"- Label '{label}': {MAIN_LABEL_GUIDELINES[label]}")
                elif label in LABEL_NAME_TO_ID:
                    lid = LABEL_NAME_TO_ID[label]
                    if lid in MAIN_LABEL_GUIDELINES:
                        guidelines.append(f"- Label '{label}' ({lid}): {MAIN_LABEL_GUIDELINES[lid]}")
            if guidelines:
                guidelines_text = "Main Label Guidelines:\n" + "\n".join(guidelines)
        
        context_block = ""
        if available_sub_labels_context:
            content = "\n".join(available_sub_labels_context) if isinstance(available_sub_labels_context, list) else str(available_sub_labels_context)
            context_block = f"Available specific sub-labels:\n{content}"

        descriptions_text = "\n".join(plot_descriptions) if plot_descriptions else ""

        # Adjust prompt based on whether graphs are included
        if include_graphs and images:
            intro_text = "Analyze the provided telemetry data graphs for a racing simulation segment."
            task_text = """Task:
1. Examine the telemetry graphs and trajectory map.
2. Identify driving maneuvers, incidents, or patterns.
3. Recommend specific labels/sub-labels from the provided list.
4. Return the result STRICTLY as a JSON object."""
        else:
            intro_text = "Analyze the following racing simulation segment based on the label context provided."
            task_text = """Task:
1. Based on the current labels and guidelines provided, recommend appropriate sub-labels.
2. Use the label hierarchy and guidelines to make informed suggestions.
3. Return the result STRICTLY as a JSON object."""

        prompt_text = f"""
{intro_text}
{descriptions_text}

{labels_context}
{guidelines_text}

{context_block}

{task_text}

JSON Schema:
{{
    "analysis_summary": "Short text summary of the analysis.",
    "suggested_labels": [
        {{
            "label": "Name of the label (e.g., 'Corner Entry Overspeed')",
            "confidence": 0.0 to 1.0,
            "reasoning": "Why this label applies based on the graph data."
        }}
    ],
    "primary_issue_detected": true/false
}}
"""
        # Build content parts for batch API file format
        parts = [{"text": prompt_text}]
        
        # Add images as inline base64 data
        for img in images:
            b64_data = self._image_to_base64(img, "PNG")
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": b64_data
                }
            })
        
        # Return request in Gemini Batch API file format (JSONL line)
        return {
            "key": f"segment-{segment_index}",
            "request": {
                "contents": [{"parts": parts}],
                "generation_config": {"response_mime_type": "application/json"}
            }
        }

    def submit_batch_file_job(
        self, 
        requests_list: List[Dict[str, Any]],
        display_name: str = "acla-batch-annotation-job",
        gcs_bucket: Optional[str] = None,
        gcp_project: Optional[str] = None
    ) -> Optional[Any]:
        """
        Submits a file-based batch job to Gemini Batch API.
        
        IMPORTANT: The Gemini Batch API requires Google Cloud Storage (GCS) or BigQuery
        as the source. This method uploads the JSONL file to GCS.
        
        Authentication: Uses Google Application Default Credentials (ADC).
        Set up ADC by one of:
          1. Run: gcloud auth application-default login
          2. Set GOOGLE_APPLICATION_CREDENTIALS env var to a service account key file
          3. Use a GCP compute environment with attached service account
        
        Required GCP IAM permissions:
          - storage.objects.create (to upload batch input file)
          - storage.objects.get (to read batch results)
          - aiplatform.batchPredictionJobs.create (or Gemini equivalent)
        
        1. Creates a JSONL file from requests_list
        2. Uploads the file to GCS bucket
        3. Creates batch job referencing the GCS URI
        
        Args:
            requests_list: List of request dicts with 'key' and 'request' fields
            display_name: Job display name for tracking
            gcs_bucket: GCS bucket name (without gs:// prefix). 
                       Can also be set via GEMINI_BATCH_GCS_BUCKET env var.
            gcp_project: GCP project ID. If None, uses ADC default project or
                        GOOGLE_CLOUD_PROJECT env var.
            
        Returns:
            BatchJob object or None on failure
        """
        import tempfile
        import os as os_module
        
        if not requests_list:
            print("DEBUG: No requests to submit for batch job.")
            return None
        
        # Get GCS bucket from param or environment
        bucket_name = gcs_bucket or os_module.environ.get("GEMINI_BATCH_GCS_BUCKET")
        if not bucket_name:
            print("DEBUG: ERROR - GCS bucket required for Gemini Batch API.")
            print("DEBUG: Set GEMINI_BATCH_GCS_BUCKET env var or pass gcs_bucket parameter.")
            print("DEBUG: The Gemini Batch API only supports gs:// or bq:// sources.")
            return None
        
        # Get GCP project from param or environment
        project_id = gcp_project or os_module.environ.get("GOOGLE_CLOUD_PROJECT")
            
        try:
            from google.cloud import storage
        except ImportError:
            print("DEBUG: ERROR - google-cloud-storage package required for batch jobs.")
            print("DEBUG: Install with: pip install google-cloud-storage")
            return None
            
        try:
            # 1. Create JSONL file locally
            jsonl_filename = f"{display_name}_{int(time.time())}.jsonl"
            temp_dir = tempfile.gettempdir()
            jsonl_path = os_module.path.join(temp_dir, jsonl_filename)
            
            print(f"DEBUG: Writing {len(requests_list)} requests to JSONL file: {jsonl_path}")
            
            with open(jsonl_path, "w") as f:
                for req in requests_list:
                    f.write(json.dumps(req) + "\n")
            
            file_size_kb = os_module.path.getsize(jsonl_path) / 1024
            print(f"DEBUG: JSONL file size: {file_size_kb:.1f} KB")
            
            # 2. Upload file to GCS using Application Default Credentials
            # ADC is automatically used by storage.Client() when no explicit credentials provided
            # To set up ADC: gcloud auth application-default login
            # Or set GOOGLE_APPLICATION_CREDENTIALS to service account key path
            print(f"DEBUG: Uploading JSONL file to GCS bucket: {bucket_name}")
            print("DEBUG: Using Application Default Credentials (ADC) for GCS authentication")
            try:
                storage_client = storage.Client(project=project_id)
            except Exception as auth_err:
                print(f"DEBUG: GCS authentication failed: {auth_err}")
                print("DEBUG: Ensure ADC is configured:")
                print("DEBUG:   1. Run: gcloud auth application-default login")
                print("DEBUG:   2. Or set GOOGLE_APPLICATION_CREDENTIALS env var")
                return None
            
            bucket = storage_client.bucket(bucket_name)
            
            gcs_blob_path = f"gemini-batch/{jsonl_filename}"
            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_filename(jsonl_path)
            
            gcs_uri = f"gs://{bucket_name}/{gcs_blob_path}"
            print(f"DEBUG: File uploaded to: {gcs_uri}")
            
            # 3. Create batch job with GCS URI
            print(f"DEBUG: Creating batch job with {len(requests_list)} requests...")
            try:
                batch_job = self.client.batches.create(
                    model=self.model_name,
                    src=gcs_uri,
                    config={'display_name': display_name}
                )
            except Exception as batch_err:
                import traceback
                print(f"DEBUG: Failed to create batch job: {batch_err}")
                print(f"DEBUG: Detailed stack trace:\n{traceback.format_exc()}")
                return None
            
            print(f"DEBUG: Created batch job: {batch_job.name}")
            
            # Clean up temp file
            try:
                os_module.remove(jsonl_path)
            except:
                pass
                
            return batch_job
            
        except Exception as e:
            import traceback
            print(f"DEBUG: Failed to create batch job: {e}")
            print(f"DEBUG: Detailed stack trace:\n{traceback.format_exc()}")
            return None

    def poll_batch_job_status(self, job_name: str, poll_interval: float = 30.0, timeout: float = 3600.0) -> Optional[Any]:
        """
        Polls batch job status until completion or timeout.
        
        Args:
            job_name: The batch job name (e.g., 'batches/xyz...')
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait in seconds (default 1 hour)
            
        Returns:
            Final BatchJob object or None on timeout/error
        """
        import time as time_module
        start_time = time_module.time()
        
        while True:
            try:
                batch_job = self.client.batches.get(name=job_name)
                state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
                
                print(f"DEBUG: Batch job '{job_name}' state: {state_name}")
                
                if state_name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
                    return batch_job
                    
                elapsed = time_module.time() - start_time
                if elapsed > timeout:
                    print(f"DEBUG: Batch job polling timed out after {elapsed:.1f}s")
                    return None
                    
                time_module.sleep(poll_interval)
                
            except Exception as e:
                print(f"DEBUG: Error polling batch job status: {e}")
                return None

    def get_batch_job_status(self, job_name: str) -> Dict[str, Any]:
        """
        Gets the current status of a batch job without blocking.
        
        Returns:
            Dict with status info: {state, name, create_time, finished, error}
        """
        try:
            batch_job = self.client.batches.get(name=job_name)
            state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
            
            finished_states = ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED')
            
            return {
                "state": state_name,
                "name": batch_job.name,
                "create_time": str(batch_job.create_time) if hasattr(batch_job, 'create_time') else None,
                "finished": state_name in finished_states,
                "success": state_name == 'JOB_STATE_SUCCEEDED',
                "error": None
            }
        except Exception as e:
            return {
                "state": "ERROR",
                "name": job_name,
                "finished": True,
                "success": False,
                "error": str(e)
            }

    def process_batch_file_results(self, batch_job: Any) -> List[Dict[str, Any]]:
        """
        Processes results from a completed file-based batch job.
        Downloads the result JSONL file and parses each line.
        
        Args:
            batch_job: The completed BatchJob object
            
        Returns:
            List of parsed results with 'key' and response data
        """
        results = []
        
        try:
            state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
            
            if state_name != 'JOB_STATE_SUCCEEDED':
                print(f"DEBUG: Batch job did not succeed. State: {state_name}")
                return results
                
            # For file jobs, results are in dest.file_name
            if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                result_file_name = batch_job.dest.file_name
                print(f"DEBUG: Downloading results from file: {result_file_name}")
                
                # Download the result file
                file_content_bytes = self.client.files.download(file=result_file_name)
                file_content = file_content_bytes.decode('utf-8')
                
                # Parse JSONL - each line is a result
                for line_num, line in enumerate(file_content.splitlines()):
                    if not line.strip():
                        continue
                        
                    result_entry = {
                        "index": line_num,
                        "key": None,
                        "response_json": None,
                        "error": None
                    }
                    
                    try:
                        parsed_line = json.loads(line)
                        
                        # Extract key from response
                        result_entry["key"] = parsed_line.get("key")
                        
                        # Check for response or error
                        if "response" in parsed_line and parsed_line["response"]:
                            response_data = parsed_line["response"]
                            
                            # Try to extract text from candidates
                            if "candidates" in response_data:
                                for candidate in response_data["candidates"]:
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            if "text" in part:
                                                result_entry["response_json"] = part["text"]
                                                break
                                        if result_entry["response_json"]:
                                            break
                                            
                        elif "error" in parsed_line:
                            result_entry["error"] = str(parsed_line["error"])
                        else:
                            result_entry["error"] = "No response or error in result line"
                            
                    except json.JSONDecodeError as e:
                        result_entry["error"] = f"JSON parse error on line {line_num}: {e}"
                        
                    results.append(result_entry)
                    
                print(f"DEBUG: Processed {len(results)} results from file.")
                    
            else:
                print("DEBUG: No file_name found in batch job dest.")
                
        except Exception as e:
            print(f"DEBUG: Error processing batch results: {e}")
            
        return results

    def cancel_batch_job(self, job_name: str) -> bool:
        """
        Cancels a running batch job.
        
        Returns:
            True if cancellation was successful
        """
        try:
            self.client.batches.cancel(name=job_name)
            print(f"DEBUG: Batch job '{job_name}' cancellation requested.")
            return True
        except Exception as e:
            print(f"DEBUG: Failed to cancel batch job: {e}")
            return False
