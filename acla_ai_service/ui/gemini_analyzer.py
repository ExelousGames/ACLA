import os
import time
import io
import base64
import threading
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
    from app.models.segment_models import MAIN_LABEL_GUIDELINES, LABEL_NAME_TO_ID
except ImportError:
    # Fallback or mock if app module is not in path for standalone testing
    MAIN_LABEL_GUIDELINES = {}
    LABEL_NAME_TO_ID = {}
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
    }
]

class GeminiAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not api_key:
            raise ValueError("API Key is required")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-3.1-pro-preview'  # or the latest available Gemini model

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

    def create_context_trajectory_plot(self, df: pd.DataFrame, track_config: Dict[str, str], context_df: Optional[pd.DataFrame] = None) -> Optional[Image.Image]:
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
            if cpx in context_df.columns and cpy in context_df.columns:
                 ax.plot(context_df[cpx], context_df[cpy], label="Full Track", color="lightgray", linewidth=1.5, linestyle="-")
        
        # Player Segment
        if "player_x" in track_config and "player_y" in track_config:
            px = track_config["player_x"]
            py = track_config["player_y"]
            if px in df.columns and py in df.columns:
                ax.plot(df[px], df[py], label="Segment", color="green", linewidth=3)
                # Mark start and end
                ax.scatter(df[px].iloc[0], df[py].iloc[0], marker='x', color='black', label='Start')
                ax.scatter(df[px].iloc[-1], df[py].iloc[-1], marker='o', color='black', label='End')

        ax.set_title("Track Location Context")
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
                # Mark start and end
                ax.scatter(df[px].iloc[0], df[py].iloc[0], marker='x', color='green', label='Start')
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
                    ax.scatter(apex_data["x"], apex_data["y"], marker='*', s=200, color='purple', label='Min Speed Apex', zorder=5)
                    # Annotate speed if available
                    if "Speed" in apex_data and apex_data["Speed"] is not None:
                        ax.annotate(f"Min: {apex_data['Speed']:.1f}", (apex_data["x"], apex_data["y"]), xytext=(10, 10), textcoords='offset points', fontsize=10, color='purple', fontweight='bold')

        # Expert (Reference)
        if "expert_x" in track_config and "expert_y" in track_config:
            ex = track_config["expert_x"]
            ey = track_config["expert_y"]
            if ex in df.columns and ey in df.columns:
                 ax.plot(df[ex], df[ey], label="Expert Line", color="blue", linestyle="--", alpha=0.6)

        # Opponents (Optional, kept for completeness but low alpha)
        for i in range(1, 6):
            ox = f"Opponent_{i}_pos_x"
            oy = f"Opponent_{i}_pos_y"
            if ox in df.columns and oy in df.columns:
                 if df[ox].abs().max() > 0.1 or df[oy].abs().max() > 0.1:
                    ax.plot(df[ox], df[oy], label=f"Opponent {i}", color="red", alpha=0.3)

        ax.set_title("Detailed Segment Trajectory")
        ax.legend()
        ax.set_aspect('equal', 'box')
        ax.grid(True)

        return self._plot_to_image(fig)

    def _prepare_visual_content(self, df: pd.DataFrame, graph_definitions: List[Dict[str, Any]], track_config: Dict[str, str] = {}, context_df: pd.DataFrame = None) -> tuple[List[Image.Image], List[str]]:
        """Helper to generate images and descriptions shared between analysis methods."""
        images = []
        plot_descriptions = []
        
        graphs_to_process = graph_definitions if graph_definitions else GRAPH_DEFINITIONS

        for graph_def in graphs_to_process:
            cols = graph_def.get("columns", [])
            title = graph_def.get("title", f"Graph {graph_def.get('id')}")
            description = graph_def.get("description", "")
            
            valid_cols = [c for c in columns if c in df.columns] if 'columns' in locals() else [c for c in cols if c in df.columns] # Fix for variable check
            # Correct logic:
            valid_cols = [c for c in cols if c in df.columns]

            if valid_cols:
                img = self.create_feature_plot(df, valid_cols, title)
                if img:
                    images.append(img)
                    desc_text = f"Image {len(images)}: {title}. Visualizes {', '.join(valid_cols)}."
                    if description:
                        desc_text += f"\n   Context/Meaning: {description}"
                    plot_descriptions.append(desc_text)

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
            # 1. Big Picture Context
            if context_df is not None and not context_df.empty:
                ctx_img = self.create_context_trajectory_plot(df, track_config, context_df=context_df)
                if ctx_img:
                    images.append(ctx_img)
                    plot_descriptions.append(f"Image {len(images)}: Track Map Overview. Shows where the segment (green) is located on the full track (grey).")

            # 2. Detailed Trajectory
            traj_img = self.create_detailed_trajectory_plot(df, track_config)
            if traj_img:
                images.append(traj_img)
                plot_descriptions.append(f"Image {len(images)}: Detailed Trajectory. Close-up of path, showing corner apex and minimum speed points.")
        
        return images, plot_descriptions

    def analyze_segment_json(self, 
                        df: pd.DataFrame, 
                        graph_definitions: List[Dict[str, Any]] = None, 
                        track_config: Dict[str, str] = {},
                        current_labels: List[str] = None,
                        available_sub_labels_context: List[str] = None,
                        context_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyzes the segment and returns a JSON structured response.
        Useful for batch processing / auto-annotation.
        """
        images, plot_descriptions = self._prepare_visual_content(df, graph_definitions, track_config, context_df)

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
                        context_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyzes the segment and returns a dictionary with:
        - 'response': The text response from Gemini
        - 'images': List of PIL Images generated and sent
        - 'prompt': The text prompt sent
        """
        images, plot_descriptions = self._prepare_visual_content(df, graph_definitions, track_config, context_df)

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
            context_block = f"""
            Available specific sub-labels based on current selection:
            {content}
            """

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
