# Standard Telemetry Fields

## Contract

This is the application-wide telemetry standard. Its initial complete key set is the flat contract produced by the current `src/py-scripts/ACCMemoryExtractor.py` after flattening the data classes exposed by [PyAccSharedMemory](https://github.com/rrennoir/PyAccSharedMemory). These exact names are already consumed by the application and accepted by the upload path.

- Every game uses this same standard. A game-specific reader maps its SDK/shared-memory fields to these names and types before it emits a sample. A reader omits fields its game cannot supply; it must not create game-prefixed variants, aliases, or additional telemetry names.
- Field meanings, coordinate conventions, and units follow the originating PyAccSharedMemory contract. A reader converts its raw values when necessary to meet those standard semantics before emitting the sample; downstream components do not perform game-specific conversions.
- Historical names remain stable even when they originated with ACC. A future reader uses the semantically equivalent standard field or omits it; it does not introduce a clearer replacement name.
- A successful recorded row is one flat JSON object containing only keys from this catalog, regardless of the source game.
- After a reader has produced the standard object, the writer, saved-file reader, renderer, and upload path preserve every telemetry key and value unchanged. They must not rename fields, add aliases, convert units, wrap the row in another persisted object, or add metadata fields to the row.
- The standard contains 240 keys total: 133 Physics, 84 Graphics, and 23 Static. Adding a new game does not extend or rename this list.
- `Graphics_status`, `Graphics_session_type`, `Graphics_flag`, `Graphics_penalty`, `Graphics_track_grip_status`, and the three `Graphics_rain_intensity*` fields contain the integer `.value` of their PyAccSharedMemory enum.
- Preserve runtime values even where PyAccSharedMemory's annotations and assignments differ: the current library populates `Graphics_last_sector_time_str` with the integer `lastSectorTime` value and leaves `Graphics_rain_tyres` as an integer `0`/`1`. The pipeline must not coerce or rename either field.
- `Graphics_car_coordinates` remains an array of 60 `{ "x": number, "y": number, "z": number }` objects. `Graphics_car_id` remains an array of 60 integers. Every other nested PyAccSharedMemory data class is flattened into the exact keys below.
- A cataloged key can be absent from an individual row when its source game cannot supply it or its reader treats its value as unavailable. Absence does not authorize a replacement name.
- `{"available":false}` is an ACC reader-control message, not telemetry. It must never be written to the recording or uploaded. Other readers likewise keep their control messages outside the standard telemetry object.

Sources: [PyAccSharedMemory data classes](https://github.com/rrennoir/PyAccSharedMemory/blob/main/src/pyaccsharedmemory.py), [PyAccSharedMemory field documentation](https://github.com/rrennoir/PyAccSharedMemory#dataclass), and the current [ACCMemoryExtractor.py](../src/py-scripts/ACCMemoryExtractor.py).

## Physics fields (133)

```text
Physics_packed_id                                      integer
Physics_gas                                            number
Physics_brake                                          number
Physics_fuel                                           number
Physics_gear                                           integer
Physics_rpm                                            integer
Physics_steer_angle                                    number
Physics_speed_kmh                                      number
Physics_velocity_x                                     number
Physics_velocity_y                                     number
Physics_velocity_z                                     number
Physics_g_force_x                                      number
Physics_g_force_y                                      number
Physics_g_force_z                                      number
Physics_wheel_slip_front_left                          number
Physics_wheel_slip_front_right                         number
Physics_wheel_slip_rear_left                           number
Physics_wheel_slip_rear_right                          number
Physics_wheel_pressure_front_left                      number
Physics_wheel_pressure_front_right                     number
Physics_wheel_pressure_rear_left                       number
Physics_wheel_pressure_rear_right                      number
Physics_wheel_angular_s_front_left                     number
Physics_wheel_angular_s_front_right                    number
Physics_wheel_angular_s_rear_left                      number
Physics_wheel_angular_s_rear_right                     number
Physics_tyre_core_temp_front_left                      number
Physics_tyre_core_temp_front_right                     number
Physics_tyre_core_temp_rear_left                       number
Physics_tyre_core_temp_rear_right                      number
Physics_suspension_travel_front_left                   number
Physics_suspension_travel_front_right                  number
Physics_suspension_travel_rear_left                    number
Physics_suspension_travel_rear_right                   number
Physics_tc                                             number
Physics_heading                                        number
Physics_pitch                                          number
Physics_roll                                           number
Physics_car_damage_front                               number
Physics_car_damage_rear                                number
Physics_car_damage_left                                number
Physics_car_damage_right                               number
Physics_car_damage_center                              number
Physics_pit_limiter_on                                 boolean
Physics_abs                                            number
Physics_autoshifter_on                                 boolean
Physics_turbo_boost                                    number
Physics_air_temp                                       number
Physics_road_temp                                      number
Physics_local_angular_vel_x                            number
Physics_local_angular_vel_y                            number
Physics_local_angular_vel_z                            number
Physics_final_ff                                       number
Physics_brake_temp_front_left                          number
Physics_brake_temp_front_right                         number
Physics_brake_temp_rear_left                           number
Physics_brake_temp_rear_right                          number
Physics_clutch                                         number
Physics_is_ai_controlled                               boolean
Physics_tyre_contact_point_front_left_x                number
Physics_tyre_contact_point_front_left_y                number
Physics_tyre_contact_point_front_left_z                number
Physics_tyre_contact_point_front_right_x               number
Physics_tyre_contact_point_front_right_y               number
Physics_tyre_contact_point_front_right_z               number
Physics_tyre_contact_point_rear_left_x                 number
Physics_tyre_contact_point_rear_left_y                 number
Physics_tyre_contact_point_rear_left_z                 number
Physics_tyre_contact_point_rear_right_x                number
Physics_tyre_contact_point_rear_right_y                number
Physics_tyre_contact_point_rear_right_z                number
Physics_tyre_contact_normal_front_left_x               number
Physics_tyre_contact_normal_front_left_y               number
Physics_tyre_contact_normal_front_left_z               number
Physics_tyre_contact_normal_front_right_x              number
Physics_tyre_contact_normal_front_right_y              number
Physics_tyre_contact_normal_front_right_z              number
Physics_tyre_contact_normal_rear_left_x                number
Physics_tyre_contact_normal_rear_left_y                number
Physics_tyre_contact_normal_rear_left_z                number
Physics_tyre_contact_normal_rear_right_x               number
Physics_tyre_contact_normal_rear_right_y               number
Physics_tyre_contact_normal_rear_right_z               number
Physics_tyre_contact_heading_front_left_x              number
Physics_tyre_contact_heading_front_left_y              number
Physics_tyre_contact_heading_front_left_z              number
Physics_tyre_contact_heading_front_right_x             number
Physics_tyre_contact_heading_front_right_y             number
Physics_tyre_contact_heading_front_right_z             number
Physics_tyre_contact_heading_rear_left_x               number
Physics_tyre_contact_heading_rear_left_y               number
Physics_tyre_contact_heading_rear_left_z               number
Physics_tyre_contact_heading_rear_right_x              number
Physics_tyre_contact_heading_rear_right_y              number
Physics_tyre_contact_heading_rear_right_z              number
Physics_brake_bias                                      number
Physics_local_velocity_x                                number
Physics_local_velocity_y                                number
Physics_local_velocity_z                                number
Physics_slip_ratio_front_left                           number
Physics_slip_ratio_front_right                          number
Physics_slip_ratio_rear_left                            number
Physics_slip_ratio_rear_right                           number
Physics_slip_angle_front_left                           number
Physics_slip_angle_front_right                          number
Physics_slip_angle_rear_left                            number
Physics_slip_angle_rear_right                           number
Physics_suspension_damage_front_left                    number
Physics_suspension_damage_front_right                   number
Physics_suspension_damage_rear_left                     number
Physics_suspension_damage_rear_right                    number
Physics_water_temp                                      number
Physics_brake_pressure_front_left                       number
Physics_brake_pressure_front_right                      number
Physics_brake_pressure_rear_left                        number
Physics_brake_pressure_rear_right                       number
Physics_front_brake_compound                            integer
Physics_rear_brake_compound                             integer
Physics_pad_life_front_left                             number
Physics_pad_life_front_right                            number
Physics_pad_life_rear_left                              number
Physics_pad_life_rear_right                             number
Physics_disc_life_front_left                            number
Physics_disc_life_front_right                           number
Physics_disc_life_rear_left                             number
Physics_disc_life_rear_right                            number
Physics_ignition_on                                     boolean
Physics_starter_engine_on                               boolean
Physics_is_engine_running                               boolean
Physics_kerb_vibration                                  number
Physics_slip_vibration                                  number
Physics_g_vibration                                     number
Physics_abs_vibration                                   number
```

## Graphics fields (84)

```text
Graphics_packed_id                                      integer
Graphics_status                                         integer (ACC_STATUS value)
Graphics_session_type                                   integer (ACC_SESSION_TYPE value)
Graphics_current_time_str                               string
Graphics_last_time_str                                  string
Graphics_best_time_str                                  string
Graphics_last_sector_time_str                           integer (current PyAccSharedMemory runtime value)
Graphics_completed_lap                                  integer
Graphics_position                                       integer
Graphics_current_time                                   integer
Graphics_last_time                                      integer
Graphics_best_time                                      integer
Graphics_session_time_left                              number
Graphics_distance_traveled                              number
Graphics_is_in_pit                                      boolean
Graphics_current_sector_index                           integer
Graphics_last_sector_time                               integer
Graphics_number_of_laps                                 integer
Graphics_tyre_compound                                  string
Graphics_normalized_car_position                        number
Graphics_active_cars                                    integer
Graphics_car_coordinates                                array<{x: number, y: number, z: number}>[60]
Graphics_car_id                                         integer[60]
Graphics_player_car_id                                  integer
Graphics_penalty_time                                   number
Graphics_flag                                           integer (ACC_FLAG_TYPE value)
Graphics_penalty                                        integer (ACC_PENALTY_TYPE value)
Graphics_ideal_line_on                                  boolean
Graphics_is_in_pit_lane                                 boolean
Graphics_mandatory_pit_done                             boolean
Graphics_wind_speed                                     number
Graphics_wind_direction                                 number
Graphics_is_setup_menu_visible                          boolean
Graphics_main_display_index                             integer
Graphics_secondary_display_index                        integer
Graphics_tc_level                                       integer
Graphics_tc_cut_level                                   integer
Graphics_engine_map                                     integer
Graphics_abs_level                                      integer
Graphics_fuel_per_lap                                   number
Graphics_rain_light                                     boolean
Graphics_flashing_light                                 boolean
Graphics_light_stage                                    integer
Graphics_exhaust_temp                                   number
Graphics_wiper_stage                                    integer
Graphics_driver_stint_total_time_left                   integer
Graphics_driver_stint_time_left                         integer
Graphics_rain_tyres                                     integer (0 or 1)
Graphics_session_index                                  integer
Graphics_used_fuel                                      number
Graphics_delta_lap_time_str                             string
Graphics_delta_lap_time                                 integer
Graphics_estimated_lap_time_str                         string
Graphics_estimated_lap_time                             integer
Graphics_is_delta_positive                              boolean
Graphics_is_valid_lap                                   boolean
Graphics_fuel_estimated_laps                            number
Graphics_track_status                                   string
Graphics_missing_mandatory_pits                         integer
Graphics_clock                                          number
Graphics_direction_light_left                           boolean
Graphics_direction_light_right                          boolean
Graphics_global_yellow                                  boolean
Graphics_global_yellow_s1                               boolean
Graphics_global_yellow_s2                               boolean
Graphics_global_yellow_s3                               boolean
Graphics_global_white                                   boolean
Graphics_global_green                                   boolean
Graphics_global_chequered                               boolean
Graphics_global_red                                     boolean
Graphics_mfd_tyre_set                                   integer
Graphics_mfd_fuel_to_add                                number
Graphics_mfd_tyre_pressure_front_left                   number
Graphics_mfd_tyre_pressure_front_right                  number
Graphics_mfd_tyre_pressure_rear_left                    number
Graphics_mfd_tyre_pressure_rear_right                   number
Graphics_track_grip_status                              integer (ACC_TRACK_GRIP_STATUS value)
Graphics_rain_intensity                                 integer (ACC_RAIN_INTENSITY value)
Graphics_rain_intensity_in_10min                        integer (ACC_RAIN_INTENSITY value)
Graphics_rain_intensity_in_30min                        integer (ACC_RAIN_INTENSITY value)
Graphics_current_tyre_set                               integer
Graphics_strategy_tyre_set                              integer
Graphics_gap_ahead                                      integer
Graphics_gap_behind                                     integer
```

## Static fields (23)

```text
Static_sm_version                                       string
Static_ac_version                                       string
Static_number_of_session                                integer
Static_num_cars                                         integer
Static_car_model                                        string
Static_track                                            string
Static_player_name                                      string
Static_player_surname                                   string
Static_player_nick                                      string
Static_sector_count                                     integer
Static_max_rpm                                          integer
Static_max_fuel                                         number
Static_penalty_enabled                                  boolean
Static_aid_fuel_rate                                    number
Static_aid_tyre_rate                                    number
Static_aid_mechanical_damage                            number
Static_aid_stability                                    number
Static_aid_auto_clutch                                  boolean
Static_pit_window_start                                 integer
Static_pit_window_end                                   integer
Static_is_online                                        boolean
Static_dry_tyres_name                                   string
Static_wet_tyres_name                                   string
```

## Standard enum values

The numeric meanings below are part of the standard. The labels are the originating PyAccSharedMemory enum members, not additional telemetry fields. Every game reader maps its source values to these integers.

- `Graphics_status`: `0` ACC_OFF, `1` ACC_REPLAY, `2` ACC_LIVE, `3` ACC_PAUSE.
- `Graphics_session_type`: `-1` ACC_UNKNOW, `0` ACC_PRACTICE, `1` ACC_QUALIFY, `2` ACC_RACE, `3` ACC_HOTLAP, `4` ACC_TIME_ATTACK, `5` ACC_DRIFT, `6` ACC_DRAG, `7` ACC_HOTSTINT, `8` ACC_HOTLAPSUPERPOLE.
- `Graphics_flag`: `0` ACC_NO_FLAG, `1` ACC_BLUE_FLAG, `2` ACC_YELLOW_FLAG, `3` ACC_BLACK_FLAG, `4` ACC_WHITE_FLAG, `5` ACC_CHECKERED_FLAG, `6` ACC_PENALTY_FLAG, `7` ACC_GREEN_FLAG, `8` ACC_ORANGE_FLAG.
- `Graphics_penalty`: `-1` UnknownValue; `0` No_penalty; `1`–`6` cutting penalties; `7`–`12` pit-speeding penalties; `13` Disqualified_IgnoredMandatoryPit; `14` PostRaceTime; `15` Disqualified_Trolling; `16` Disqualified_PitEntry; `17` Disqualified_PitExit; `18` Disqualified_WrongWay_old; `19` DriveThrough_IgnoredDriverStint; `20` Disqualified_IgnoredDriverStint; `21` Disqualified_ExceededDriverStintLimit; `22` Disqualified_WrongWay.
- `Graphics_track_grip_status`: `0` ACC_GREEN, `1` ACC_FAST, `2` ACC_OPTIMUM, `3` ACC_GREASY, `4` ACC_DAMP, `5` ACC_WET, `6` ACC_FLOODED.
- Each `Graphics_rain_intensity*` field: `0` ACC_NO_RAIN, `1` ACC_DRIZZLE, `2` ACC_LIGHT_RAIN, `3` ACC_MEDIUM_RAIN, `4` ACC_HEAVY_RAIN, `5` ACC_THUNDERSTORM.
