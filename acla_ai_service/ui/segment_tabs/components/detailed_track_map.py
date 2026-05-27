import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app.domain.telemetry import MAX_CARS

def render_track_map(df, viz_start_idx, viz_end_idx, session_id):
    st.subheader("Track Map & Positions")

    # Check if we have position data
    has_player_pos = "Graphics_player_pos_x" in df.columns and "Graphics_player_pos_y" in df.columns
    has_player_pos_z = "Graphics_player_pos_z" in df.columns

    has_opponent_pos = any(f"Car_{i}_pos_x" in df.columns for i in range(1, MAX_CARS + 1))
    
    has_expert_pos = "expert_optimal_player_pos_x" in df.columns and "expert_optimal_player_pos_y" in df.columns
    has_expert_pos_z = "expert_optimal_player_pos_z" in df.columns
    
    if has_player_pos or has_opponent_pos or has_expert_pos:
        # View controls
        st.caption("Axis Settings")
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5, col_ctrl6 = st.columns(6)
        with col_ctrl1:
            invert_x = st.checkbox("Invert X", value=st.session_state.get("saved_detailed_invert_x", False), key="detailed_invert_x")
            st.session_state.saved_detailed_invert_x = invert_x
        with col_ctrl2:
            invert_y = st.checkbox("Invert Y", value=st.session_state.get("saved_detailed_invert_y", False), key="detailed_invert_y")
            st.session_state.saved_detailed_invert_y = invert_y
        with col_ctrl3:
            invert_z = st.checkbox("Invert Z", value=st.session_state.get("saved_detailed_invert_z", False), key="detailed_invert_z")
            st.session_state.saved_detailed_invert_z = invert_z
        with col_ctrl4:
            only_player = st.checkbox("Only Player", value=st.session_state.get("saved_detailed_only_player", False), key="detailed_only_player")
            st.session_state.saved_detailed_only_player = only_player
        with col_ctrl5:
            only_closest = st.checkbox("Only 5 closest cars", value=st.session_state.get("saved_detailed_only_closest", False), key="detailed_only_closest")
            st.session_state.saved_detailed_only_closest = only_closest
        with col_ctrl6:
            traj_opts = ["Gas/Brake", "Balance (Oversteer/Understeer)", "Solid Green"]
            saved_traj = st.session_state.get("saved_detailed_traj_color_mode", "Gas/Brake")
            traj_color_mode = st.selectbox(
                "Trajectory Color",
                traj_opts,
                index=traj_opts.index(saved_traj) if saved_traj in traj_opts else 0,
                key="detailed_traj_color_mode",
                help="Gas/Brake: Green = Gas, Red = Brake.\n\nBalance: Red = Oversteer, Blue = Understeer."
            )
            st.session_state.saved_detailed_traj_color_mode = traj_color_mode
        
        # Create windowed dataframe for trajectory plotting using Global Range
        start_idx = min(viz_start_idx, len(df) - 1)
        
        # Ensure indices are within bounds (include end index in slice)
        safe_end_idx = min(viz_end_idx + 1, len(df))
        
        context_plot_df = pd.DataFrame(columns=df.columns)

        if safe_end_idx <= start_idx:
             # Handle empty range selection gracefully
             map_plot_df = pd.DataFrame(columns=df.columns)
             selected_time_idx = start_idx
        else:
             map_plot_df = df.iloc[start_idx:safe_end_idx]
             selected_time_idx = min(viz_end_idx, len(df) - 1)
             
             # Calculate extended range for context
             segment_len = safe_end_idx - start_idx
             padding = max(100, int(segment_len * 0.5)) 
             ext_start_idx = max(0, start_idx - padding)
             ext_end_idx = min(len(df), safe_end_idx + padding)
             context_plot_df = df.iloc[ext_start_idx:ext_end_idx]

        current_row = df.iloc[selected_time_idx]
        start_row = df.iloc[start_idx]
        map_data = []
        
        # Helper for Max Curvature Calculation
        def get_max_curvature_point(df_in, x_col, y_col, z_col=None, speed_col=None, label_type="Player"):
            if df_in.empty or len(df_in) <= 5:
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
                        "Index": target_idx,
                        "Speed": target_row[speed_col] if speed_col and speed_col in df_in.columns else None
                    }
                    if z_col and z_col in df_in.columns:
                        p_geo["z"] = target_row[z_col]
                    return p_geo
            except Exception:
                pass
            return None

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

            # Max Curvature Point
            max_curvature_data = get_max_curvature_point(
                map_plot_df, 
                "Graphics_player_pos_x", 
                "Graphics_player_pos_y", 
                "Graphics_player_pos_z" if has_player_pos_z else None,
                "Physics_speed_kmh",
                "Player"
            )
            if max_curvature_data:
                map_data.append(max_curvature_data)

        # Add Expert Position
        if has_expert_pos and not only_player:
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

            # Expert Max Curvature Point
            e_max_curvature_data = get_max_curvature_point(
                map_plot_df,
                "expert_optimal_player_pos_x",
                "expert_optimal_player_pos_y",
                "expert_optimal_player_pos_z" if has_expert_pos_z else None,
                "expert_optimal_speed",  # Use dedicated column
                "Expert"
            )
            if e_max_curvature_data:
                map_data.append(e_max_curvature_data)
        
        # Identify the player's own slot so we don't draw ourselves as an opponent
        player_slot = None
        if has_player_pos:
            p_x = current_row["Graphics_player_pos_x"]
            p_y = current_row["Graphics_player_pos_y"]
            for i in range(1, MAX_CARS + 1):
                cx_col = f"Car_{i}_pos_x"
                cy_col = f"Car_{i}_pos_y"
                if cx_col in df.columns and cy_col in df.columns:
                    if current_row[cx_col] == p_x and current_row[cy_col] == p_y:
                        player_slot = i
                        break

        # Optionally restrict to the 5 cars closest to the player at the end index
        allowed_slots = None
        if only_closest and has_player_pos:
            candidates = []
            for i in range(1, MAX_CARS + 1):
                if i == player_slot:
                    continue
                cx_col = f"Car_{i}_pos_x"
                cy_col = f"Car_{i}_pos_y"
                if cx_col not in df.columns or cy_col not in df.columns:
                    continue
                vx = current_row[cx_col]
                vy = current_row[cy_col]
                if vx == 0 and vy == 0:
                    continue
                dist2 = (vx - p_x) ** 2 + (vy - p_y) ** 2
                candidates.append((dist2, i))
            candidates.sort()
            allowed_slots = {i for _, i in candidates[:5]}

        # Add Opponent Positions
        if not only_player:
            for i in range(1, MAX_CARS + 1):
                if i == player_slot:
                    continue
                if allowed_slots is not None and i not in allowed_slots:
                    continue
                opp_x_col = f"Car_{i}_pos_x"
                opp_y_col = f"Car_{i}_pos_y"
                opp_z_col = f"Car_{i}_pos_z"

                if opp_x_col in df.columns and opp_y_col in df.columns:
                    # Filter out inactive opponents (usually 0,0 coordinates)
                    if current_row[opp_x_col] != 0 or current_row[opp_y_col] != 0:
                        o_data = {
                            "x": current_row[opp_x_col],
                            "y": current_row[opp_y_col],
                            "Type": "Opponent",
                            "ID": f"Car {i}",
                            "Marker": "End",
                            "Index": selected_time_idx
                        }
                        if opp_z_col in df.columns:
                            o_data["z"] = current_row[opp_z_col]
                        map_data.append(o_data)
        
        if map_data:
            map_df = pd.DataFrame(map_data)
            
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
                symbol_map={"Start": "diamond", "End": "circle", "Max Curvature": "x"}
            )
            fig_map.update_traces(marker=dict(size=5))
            
            scene_dict = dict(
                aspectmode='data',
                # dragmode='turntable',
                camera=dict(
                    projection=dict(type='orthographic'),
                    up=dict(x=0, y=0, z=1)  # Fix Z-axis as up for easier yaw rotation
                )
            )
            if invert_x: scene_dict['xaxis'] = dict(autorange="reversed")
            if invert_y: scene_dict['yaxis'] = dict(autorange="reversed")
            if invert_z: scene_dict['zaxis'] = dict(autorange="reversed")
            fig_map.update_layout(scene=scene_dict, dragmode='turntable')

            # Add Trajectories
            # Player
            if has_player_pos:
                if traj_color_mode == "Gas/Brake":
                    player_ctx_color = (context_plot_df["Physics_gas"] - context_plot_df["Physics_brake"]) if "Physics_gas" in context_plot_df.columns and "Physics_brake" in context_plot_df.columns else "green"
                    player_seg_color = (map_plot_df["Physics_gas"] - map_plot_df["Physics_brake"]) if "Physics_gas" in map_plot_df.columns and "Physics_brake" in map_plot_df.columns else "green"
                    p_cmin, p_cmax, p_cscale = -1, 1, "RdYlGn"
                elif traj_color_mode == "Balance (Oversteer/Understeer)":
                    understeer_amplifier = 2.0
                    if "Physics_slip_angle_rear_left" in context_plot_df.columns and "Physics_slip_angle_front_left" in context_plot_df.columns:
                        ctx_bal = ((context_plot_df["Physics_slip_angle_rear_left"].abs() + context_plot_df["Physics_slip_angle_rear_right"].abs()) / 2) - ((context_plot_df["Physics_slip_angle_front_left"].abs() + context_plot_df["Physics_slip_angle_front_right"].abs()) / 2)
                        player_ctx_color = np.where(ctx_bal < 0, ctx_bal * understeer_amplifier, ctx_bal)
                    else:
                        player_ctx_color = "green"
                    if "Physics_slip_angle_rear_left" in map_plot_df.columns and "Physics_slip_angle_front_left" in map_plot_df.columns:
                        seg_bal = ((map_plot_df["Physics_slip_angle_rear_left"].abs() + map_plot_df["Physics_slip_angle_rear_right"].abs()) / 2) - ((map_plot_df["Physics_slip_angle_front_left"].abs() + map_plot_df["Physics_slip_angle_front_right"].abs()) / 2)
                        player_seg_color = np.where(seg_bal < 0, seg_bal * understeer_amplifier, seg_bal)
                    else:
                        player_seg_color = "green"
                    p_cmin, p_cmax, p_cscale = -0.1, 0.1, "RdBu_r"
                else:
                    player_ctx_color, player_seg_color = "green", "green"
                    p_cmin, p_cmax, p_cscale = -1, 1, "RdYlGn"

                # Add Context (Extended Trajectory) first so it renders below
                if not context_plot_df.empty:
                    if has_player_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=context_plot_df["Graphics_player_pos_x"], 
                            y=context_plot_df["Graphics_player_pos_y"],
                            z=context_plot_df["Graphics_player_pos_z"],
                            customdata=context_plot_df.index,
                            hovertemplate="Index: %{customdata}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                            mode="lines",
                            name=f"Player (Context) [{traj_color_mode}]",
                            line=dict(color=player_ctx_color, colorscale=p_cscale, cmin=p_cmin, cmax=p_cmax, width=3),
                            opacity=0.3,
                            showlegend=True
                        ))

                # Add Current Segment Trajectory
                if has_player_pos_z:
                    fig_map.add_trace(go.Scatter3d(
                        x=map_plot_df["Graphics_player_pos_x"], 
                        y=map_plot_df["Graphics_player_pos_y"],
                        z=map_plot_df["Graphics_player_pos_z"],
                        customdata=map_plot_df.index,
                        hovertemplate="Index: %{customdata}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                        mode="lines",
                        name=f"Player Trajectory [{traj_color_mode}]",
                        line=dict(color=player_seg_color, colorscale=p_cscale, cmin=p_cmin, cmax=p_cmax, width=5),
                        opacity=1.0,
                        showlegend=True
                    ))

            # Expert
            if has_expert_pos and not only_player:
                show_gas_brake_exp = traj_color_mode == "Gas/Brake"
                expert_ctx_color = (context_plot_df["expert_optimal_throttle"] - context_plot_df["expert_optimal_brake"]) if show_gas_brake_exp and "expert_optimal_throttle" in context_plot_df.columns and "expert_optimal_brake" in context_plot_df.columns else "blue"
                expert_seg_color = (map_plot_df["expert_optimal_throttle"] - map_plot_df["expert_optimal_brake"]) if show_gas_brake_exp and "expert_optimal_throttle" in map_plot_df.columns and "expert_optimal_brake" in map_plot_df.columns else "blue"

                # Context
                if not context_plot_df.empty:
                    if has_expert_pos_z:
                        fig_map.add_trace(go.Scatter3d(
                            x=context_plot_df["expert_optimal_player_pos_x"], 
                            y=context_plot_df["expert_optimal_player_pos_y"],
                            z=context_plot_df["expert_optimal_player_pos_z"],
                            customdata=context_plot_df.index,
                            hovertemplate="Index: %{customdata}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                            mode="lines",
                            name="Expert (Context)",
                            line=dict(color=expert_ctx_color, colorscale="RdYlGn", cmin=-1, cmax=1, width=3),
                            opacity=0.3,
                            showlegend=True
                        ))

                # Segment
                if has_expert_pos_z:
                    fig_map.add_trace(go.Scatter3d(
                        x=map_plot_df["expert_optimal_player_pos_x"], 
                        y=map_plot_df["expert_optimal_player_pos_y"],
                        z=map_plot_df["expert_optimal_player_pos_z"],
                        customdata=map_plot_df.index,
                        hovertemplate="Index: %{customdata}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                        mode="lines",
                        name="Expert Trajectory",
                        line=dict(color=expert_seg_color, colorscale="RdYlGn", cmin=-1, cmax=1, width=5),
                        opacity=1.0,
                        showlegend=True
                    ))
            
            # Opponents
            if not only_player:
                for i in range(1, MAX_CARS + 1):
                    if i == player_slot:
                        continue
                    if allowed_slots is not None and i not in allowed_slots:
                        continue
                    opp_x_col = f"Car_{i}_pos_x"
                    opp_y_col = f"Car_{i}_pos_y"
                    opp_z_col = f"Car_{i}_pos_z"

                    if opp_x_col in df.columns and opp_y_col in df.columns:
                        # Filter out inactive (0,0) points for cleaner trajectories

                        # Context
                        if not context_plot_df.empty:
                            opp_ctx = context_plot_df[(context_plot_df[opp_x_col] != 0) | (context_plot_df[opp_y_col] != 0)]
                            if not opp_ctx.empty:
                                if opp_z_col in df.columns:
                                    fig_map.add_trace(go.Scatter3d(
                                        x=opp_ctx[opp_x_col],
                                        y=opp_ctx[opp_y_col],
                                        z=opp_ctx[opp_z_col],
                                        customdata=opp_ctx.index,
                                        hovertemplate="Index: %{customdata}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                                        mode="lines",
                                        name=f"Car {i} (Context)",
                                        line=dict(color="aqua", width=3),
                                        opacity=0.3,
                                        showlegend=True
                                    ))

                        # Segment
                        opp_df = map_plot_df[(map_plot_df[opp_x_col] != 0) | (map_plot_df[opp_y_col] != 0)]
                        if not opp_df.empty:
                            if opp_z_col in df.columns:
                                fig_map.add_trace(go.Scatter3d(
                                    x=opp_df[opp_x_col],
                                    y=opp_df[opp_y_col],
                                    z=opp_df[opp_z_col],
                                    customdata=opp_df.index,
                                    hovertemplate="Index: %{customdata}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                                    mode="lines",
                                    name=f"Car {i} Trajectory",
                                    line=dict(color="red", width=5),
                                    opacity=1.0,
                                    showlegend=True
                                ))
            
            fig_map.update_layout(uirevision=session_id, height=800)
            st.plotly_chart(fig_map, width='stretch')
        else:
            st.info("No active cars found at this timestamp.")
    else:
        st.info("Position data (Graphics_player_pos_x/y) not available in this dataset.")

