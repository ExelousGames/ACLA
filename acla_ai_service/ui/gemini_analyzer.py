import os
import io
import base64
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for non-interactive (headless) plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image
import streamlit as st

class GeminiAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not api_key:
            raise ValueError("API Key is required")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-3.1-pro-preview')

    @staticmethod
    def get_api_key() -> Optional[str]:
        """
        Retrieves the Gemini API key from environment variables or prompts the user via Streamlit input.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_api_key_input")
        return api_key

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

    def create_trajectory_plot(self, df: pd.DataFrame, track_config: Dict[str, str]) -> Optional[Image.Image]:
        """Creates a 2D trajectory plot (top-down view)."""
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

        # Expert
        if "expert_x" in track_config and "expert_y" in track_config:
            ex = track_config["expert_x"]
            ey = track_config["expert_y"]
            if ex in df.columns and ey in df.columns:
                 ax.plot(df[ex], df[ey], label="Expert", color="blue", linestyle="--", alpha=0.7)

        # Opponents
        for i in range(1, 6):
            ox = f"Opponent_{i}_pos_x"
            oy = f"Opponent_{i}_pos_y"
            if ox in df.columns and oy in df.columns:
                 # Check if active (not 0,0)
                 if df[ox].abs().max() > 0.1 or df[oy].abs().max() > 0.1:
                    ax.plot(df[ox], df[oy], label=f"Opponent {i}", color="red", alpha=0.5)

        ax.set_title("Trajectory / Track Map")
        ax.legend()
        ax.set_aspect('equal', 'box')
        ax.grid(True)

        return self._plot_to_image(fig)

    def analyze_segment(self, 
                        df: pd.DataFrame, 
                        graph_configs: Dict[int, List[str]], 
                        track_config: Dict[str, str],
                        current_labels: List[str] = None,
                        available_sub_labels_context: List[str] = None) -> Dict[str, Any]:
        """
        Analyzes the segment and returns a dictionary with:
        - 'response': The text response from Gemini
        - 'images': List of PIL Images generated and sent
        - 'prompt': The text prompt sent
        """
        images = []
        plot_descriptions = [] # Descriptions of plots for the prompt
        
        # 1. Feature Graphs
        for graph_id, cols in graph_configs.items():
            # Use columns present in df
            valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                img = self.create_feature_plot(df, valid_cols, f"Features: {', '.join(valid_cols)}")
                if img:
                    images.append(img)
                    plot_descriptions.append(f"Image {len(images)}: Line chart showing features: {', '.join(valid_cols)}.")

        # 2. Trajectory
        # Create a simple config for trajectory if not fully provided but columns exist
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
            traj_img = self.create_trajectory_plot(df, track_config)
            if traj_img:
                images.append(traj_img)
                plot_descriptions.append(f"Image {len(images)}: Top-down trajectory map of the segment.")

        if not images:
            return {
                "response": "No valid graphs could be generated from the data.",
                "images": [],
                "prompt": "N/A"
            }

        # Construct the detailed prompt
        labels_context = f"Current identified labels: {', '.join(current_labels)}" if current_labels else "No labels identified yet."
        
        sub_labels_list = ""
        if available_sub_labels_context:
            sub_labels_list = "\\n\\nAvailable specific sub-labels based on current selection:\\n" + "\\n".join(available_sub_labels_context)

        prompt_intro = "Analyze the following telemetry data graphs for a racing simulation segment."
        descriptions_text = "\n".join(plot_descriptions)

        prompt_text = f"""
        {prompt_intro}
        {descriptions_text}

        {labels_context}
        {sub_labels_list}
        
        Task:
        1. Examine the provided telemetry graphs and trajectory map.
        2. Identify any specific driving maneuvers, incidents, or patterns.
        3. Suggest precise 'sub-labels' or additional descriptive labels for this segment.
           If specific sub-labels are provided above, prioritise selecting from them if they match the data.
           (e.g., 'Trail Braking', 'Understeer', 'Late Apex', 'Overtake Attempt', 'Loss of Traction')
        4. Explain WHY you suggest these labels based on the visual evidence in the graphs (e.g., "Sharp drop in speed combined with high steering angle suggests...").
        5. Provide your output as a concise list of suggestions with brief reasoning.
        """
        
        # Prepare content for Gemini (interleaved text and images as supported by library, or list)
        # The python library supports list of [text, img1, img2...]
        full_content = [prompt_text] + images
        
        try:
            response = self.model.generate_content(full_content)
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
