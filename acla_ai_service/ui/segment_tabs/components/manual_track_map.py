import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_manual_track_map(df, viz_start_idx, viz_end_idx, session_id):
    # --- Track Map Visualization ---
    show_track_map = st.checkbox("Show Track Map & Positions", value=True, key="manual_track_map_visible")

    if show_track_map:
        # Check if we have position data
        has_player_pos = "Graphics_player_pos_x" in df.columns and "Graphics_player_pos_y" in df.columns
        has_player_pos_z = "Graphics_player_pos_z" in df.columns
        
        has_opponent_pos = any(f"Opponent_{i}_pos_x" in df.columns for i in range(1, 6))
        
        has_expert_pos = "expert_optimal_player_pos_x" in df.columns and "expert_optimal_player_pos_y" in df.columns
        has_expert_pos_z = "expert_optimal_player_pos_z" in df.columns
        
        if has_player_pos or has_opponent_pos or has_expert_pos:
            # View controls
            st.caption("Axis Settings")
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                invert_x = st.checkbox("Invert X", value=False, key="manual_invert_x")
            with col_ctrl2:
                invert_y = st.checkbox("Invert Y", value=False, key="manual_invert_y")
            with col_ctrl3:
                invert_z = st.checkbox("Invert Z", value=False, key="manual_invert_z")
            
            # Create windowed dataframe for trajectory plotting using Global Range
            start_idx = min(viz_start_idx, len(df) - 1)
            
            # Ensure indices are within bounds
            safe_end_idx = min(viz_end_idx, len(df))
            if safe_end_idx <= start_idx:
                 # Handle empty range selection gracefully
                 map_plot_df = pd.DataFrame(columns=df.columns)
                 selected_time_idx = start_idx
            else:
                 map_plot_df = df.iloc[start_idx:safe_end_idx]
                 selected_time_idx = safe_end_idx - 1
            current_row = df.iloc[selected_time_idx]
            start_row = df.iloc[start_idx]
            map_data = []
            
            # Add Player Position
            if has_player_pos:
                # End Position
                p_data = {
                    "x": current_row["Graphics_player_pos_x"],
                    "y": current_row["Graphics_player_pos_y"],
                    "Type": "Player",
                    "ID": "Player End",
                    "Marker": "End",
                    "Index": selected_time_idx
                }
                if has_player_pos_z:
                    p_data["z"] = current_row["Graphics_player_pos_z"]
                map_data.append(p_data)

                # Start Position
                p_start = {
                    "x": start_row["Graphics_player_pos_x"],
                    "y": start_row["Graphics_player_pos_y"],
                    "Type": "Player",
                    "ID": "Player Start",
                    "Marker": "Start",
                    "Index": start_idx
                }
                if has_player_pos_z:
                    p_start["z"] = start_row["Graphics_player_pos_z"]
                map_data.append(p_start)

            # Add Expert Position
            if has_expert_pos:
                # End Position
                e_data = {
                    "x": current_row["expert_optimal_player_pos_x"],
                    "y": current_row["expert_optimal_player_pos_y"],
                    "Type": "Expert",
                    "ID": "Expert End",
                    "Marker": "End",
                    "Index": selected_time_idx
                }
                if has_expert_pos_z:
                    e_data["z"] = current_row["expert_optimal_player_pos_z"]
                map_data.append(e_data)

                # Start Position
                e_start = {
                    "x": start_row["expert_optimal_player_pos_x"],
                    "y": start_row["expert_optimal_player_pos_y"],
                    "Type": "Expert",
                    "ID": "Expert Start",
                    "Marker": "Start",
                    "Index": start_idx
                }
                if has_expert_pos_z:
                    e_start["z"] = start_row["expert_optimal_player_pos_z"]
                map_data.append(e_start)
            
            # Add Opponent Positions
            for i in range(1, 6):
                opp_x_col = f"Opponent_{i}_pos_x"
                opp_y_col = f"Opponent_{i}_pos_y"
                opp_z_col = f"Opponent_{i}_pos_z"
                opp_id_col = f"Opponent_{i}_car_id"
                
                if opp_x_col in df.columns and opp_y_col in df.columns:
                    # Filter out inactive opponents (usually 0,0 coordinates)
                    if current_row[opp_x_col] != 0 or current_row[opp_y_col] != 0:
                        opp_id = current_row[opp_id_col] if opp_id_col in df.columns else f"Opponent {i}"
                        o_data = {
                            "x": current_row[opp_x_col],
                            "y": current_row[opp_y_col],
                            "Type": "Opponent",
                            "ID": str(opp_id),
                            "Marker": "End",
                            "Index": selected_time_idx
                        }
                        if opp_z_col in df.columns:
                            o_data["z"] = current_row[opp_z_col]
                        map_data.append(o_data)
            
            if map_data:
                map_df = pd.DataFrame(map_data)
                use_3d = "z" in map_df.columns
                
                if use_3d:
                    fig_map = px.scatter_3d(
                        map_df, 
                        x="x", 
                        y="y", 
                        z="z",
                        color="Type", 
                        symbol="Marker",
                        hover_data=["ID", "Index"],
                        title=f"Positions (Start: {start_idx}, End: {selected_time_idx}) (3D)",
                        color_discrete_map={"Player": "green", "Opponent": "red", "Expert": "blue"},
                        symbol_map={"Start": "diamond", "End": "circle"}
                    )
                    fig_map.update_traces(marker=dict(size=5))
                    
                    scene_dict = dict(
                        aspectmode='data',
                        # dragmode='turntable', # Optional: explicit rotation mode
                        camera=dict(
                            projection=dict(type='orthographic'),
                            up=dict(x=0, y=0, z=1)  # Fix Z-axis as up for easier yaw rotation
                        )
                    )
                    if invert_x: scene_dict['xaxis'] = dict(autorange="reversed")
                    if invert_y: scene_dict['yaxis'] = dict(autorange="reversed")
                    if invert_z: scene_dict['zaxis'] = dict(autorange="reversed")
                    fig_map.update_layout(scene=scene_dict, dragmode='turntable')
                else:
                    fig_map = px.scatter(
                        map_df, 
                        x="x", 
                        y="y", 
                        color="Type", 
                        symbol="Marker",
                        hover_data=["ID", "Index"],
                        title=f"Positions (Start: {start_idx}, End: {selected_time_idx})",
                        color_discrete_map={"Player": "green", "Opponent": "red", "Expert": "blue"},
                        symbol_map={"Start": "x", "End": "circle"}
                    )
                    if invert_x: fig_map.update_xaxes(autorange="reversed")
                    if invert_y: fig_map.update_yaxes(autorange="reversed")

                # Add Trajectories
                # Player
                if has_player_pos:
                    if use_3d and has_player_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=map_plot_df["Graphics_player_pos_x"], 
                            y=map_plot_df["Graphics_player_pos_y"],
                            z=map_plot_df["Graphics_player_pos_z"],
                            mode="lines",
                            name="Player Trajectory",
                            customdata=map_plot_df.index,
                            hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>Index: %{customdata}<extra></extra>',
                            line=dict(color="green", width=2),
                            opacity=0.5,
                            showlegend=True
                        ))
                    else:
                        fig_map.add_trace(go.Scatter(
                            x=map_plot_df["Graphics_player_pos_x"], 
                            y=map_plot_df["Graphics_player_pos_y"],
                            mode="lines",
                            name="Player Trajectory",
                            customdata=map_plot_df.index,
                            hovertemplate='x: %{x}<br>y: %{y}<br>Index: %{customdata}<extra></extra>',
                            line=dict(color="green", width=1, dash="dot"),
                            opacity=0.5,
                            showlegend=True
                        ))

                # Expert
                if has_expert_pos:
                    if use_3d and has_expert_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=map_plot_df["expert_optimal_player_pos_x"], 
                            y=map_plot_df["expert_optimal_player_pos_y"],
                            z=map_plot_df["expert_optimal_player_pos_z"],
                            mode="lines",
                            name="Expert Trajectory",
                            customdata=map_plot_df.index,
                            hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>Index: %{customdata}<extra></extra>',
                            line=dict(color="blue", width=2),
                            opacity=0.5,
                            showlegend=True
                        ))
                    else:
                        fig_map.add_trace(go.Scatter(
                            x=map_plot_df["expert_optimal_player_pos_x"], 
                            y=map_plot_df["expert_optimal_player_pos_y"],
                            mode="lines",
                            name="Expert Trajectory",
                            customdata=map_plot_df.index,
                            hovertemplate='x: %{x}<br>y: %{y}<br>Index: %{customdata}<extra></extra>',
                            line=dict(color="blue", width=1, dash="dot"),
                            opacity=0.5,
                            showlegend=True
                        ))
                
                # Opponents
                for i in range(1, 6):
                    opp_x_col = f"Opponent_{i}_pos_x"
                    opp_y_col = f"Opponent_{i}_pos_y"
                    opp_z_col = f"Opponent_{i}_pos_z"
                    
                    if opp_x_col in df.columns and opp_y_col in df.columns:
                        # Filter out inactive (0,0) points for cleaner trajectories
                        opp_df = map_plot_df[(map_plot_df[opp_x_col] != 0) | (map_plot_df[opp_y_col] != 0)]
                        if not opp_df.empty:
                            if use_3d and opp_z_col in df.columns:
                                fig_map.add_trace(go.Scatter3d(
                                    x=opp_df[opp_x_col], 
                                    y=opp_df[opp_y_col],
                                    z=opp_df[opp_z_col],
                                    mode="lines",
                                    name=f"Opponent {i} Trajectory",
                                    customdata=opp_df.index,
                                    hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>Index: %{customdata}<extra></extra>',
                                    line=dict(color="red", width=2),
                                    opacity=0.3,
                                    showlegend=True
                                ))
                            else:
                                fig_map.add_trace(go.Scatter(
                                    x=opp_df[opp_x_col], 
                                    y=opp_df[opp_y_col],
                                    mode="lines",
                                    name=f"Opponent {i} Trajectory",
                                    customdata=opp_df.index,
                                    hovertemplate='x: %{x}<br>y: %{y}<br>Index: %{customdata}<extra></extra>',
                                    line=dict(color="red", width=1, dash="dot"),
                                    opacity=0.3,
                                    showlegend=True
                                ))

                if not use_3d:
                    fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
                
                fig_map.update_layout(uirevision=session_id, height=800)
                st.plotly_chart(fig_map, width='stretch')
            else:
                st.info("No active cars found at this timestamp.")
        else:
            st.info("Position data (Graphics_player_pos_x/y) not available in this dataset.")
