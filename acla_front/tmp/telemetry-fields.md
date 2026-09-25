# Standard Telemetry Fields

## Motion and calculated fields

The iRacing extension implements the following explicit normalization basis for the
existing scalar motion fields. These rules are tested with synthetic SDK samples;
physical equivalence with the existing ACC reader still requires controlled
stationary, straight-line, left/right-turn, banked-track, and pitch/roll captures.
The previously listed field types alone did not specify signs or gravity behavior.
World car coordinates remain unmapped in live iRacing capture. Native recorded
files provide the geographic player-position mapping described below.

| Fields | Units and intended convention | iRacing conversion |
| --- | --- | --- |
| `Physics_local_velocity_x/y/z` | m/s, body right/up/forward | `-VelocityY`, `VelocityZ`, `VelocityX` |
| `Physics_velocity_x/y/z` | m/s, fixed track frame; x = native world -Y, y = native world Z (up), z = native world X | Rotate native local velocity by `Rz(Yaw) * Ry(Pitch) * Rx(Roll)`, then map to `(-Y, Z, X)` |
| `Physics_local_angular_vel_x/y/z` | rad/s, right-hand signed rotations about body right/up/forward | `PitchRate`, `-YawRate`, `-RollRate`; angular velocity is an axial vector, so the handedness change differs from linear velocity |
| `Physics_g_force_x/y/z` | g, body right/up/forward; includes the source accelerometer's gravity contribution | `-LatAccel`, `VertAccel`, `LongAccel`, each divided by 9.80665; no flat-track gravity subtraction |
| `Physics_heading` | rad, track-local yaw about up, wrapped to [-pi, pi]; zero follows the simulator's track coordinate origin, not geographic north | Wrapped `-Yaw` |
| `Physics_pitch`, `Physics_roll` | rad, orientation about right and forward respectively, using yaw/pitch/roll rotation order | `Pitch`, `-Roll` |

The native iRacing basis used here is forward/left/up. Missing and nonfinite
local components stay absent independently. World velocity requires all three
local components and all three orientation angles from the same live sample;
if any input is missing or nonfinite, all three world components stay absent.
The rotation acts on column vectors: roll first, then pitch, then yaw. This is
equivalent to `Ry(Physics_heading) * Rx(Physics_pitch) * Rz(Physics_roll)` applied
to standard local velocity, with right-hand rotation matrices. At zero orientation,
the local and world frames coincide. The world frame follows the track's yaw origin,
not geographic north; its axes stay fixed as the car turns. Positive native pitch
rotates forward velocity toward negative world up under this convention.
Normalization belongs in the simulator reader, never in the
renderer or recorded-file reader. See the SDK channel units in the
[reader library's generated channel reference](https://irsdk-node.bengsfort.dev/API-Reference/irsdk-node-types/interfaces/TelemetryVarList/).

| Fields | Meaning and validity rules in the iRacing reader |
| --- | --- |
| `Graphics_tyre_compound` | Fitted `PlayerTireCompound` resolved by `DriverInfo.DriverTires[].TireIndex` to `TireCompoundType`; never the pending pit selection. Unknown/missing indices are omitted. |
| `Graphics_rain_tyres` | Integer 1 for Wet/Rain, 0 for explicitly dry compound types (Dry/Slick/Soft/Medium/Hard/Qualifying). Other types, including All-Purpose, leave the Boolean interpretation unknown. |
| `Graphics_distance_traveled` | Meters in the observed stint, integrated from `Speed` with the trapezoidal rule. Starts at zero only on an observed stationary pit stall or refuel. Reversing adds traveled distance, rather than subtracting lap progress. |
| `Graphics_used_fuel` | Liters consumed since the observed stationary pit start/refuel. Requires `DriverCarIsElectric` explicitly false/0 and continuous valid fuel samples. Refueling starts a new total rather than counting as negative consumption. |
| `Graphics_fuel_per_lap` | Mean liters consumed over the latest up to five fully observed non-pit laps. Fuel at the start/finish crossings is linearly interpolated. Partial laps, laps with missing fuel, reversing, and pit visits do not contribute. Refueling clears the average. |
| `Graphics_fuel_estimated_laps` | Current liters divided by a positive observed consumption average. No estimate before the first complete observed lap; electric or unknown fuel-type metadata has no liter-based estimate. |
| `Graphics_last_sector_time`, `Graphics_last_sector_time_str` | Integer milliseconds for the last complete observed sector. Both fields have the same integer value, as required by the existing catalog. Uses ordered `SplitTimeInfo.Sectors` boundaries and interpolated `SessionTime`; a partial first sector is omitted. |
| `Graphics_gap_ahead`, `Graphics_gap_behind` | Nonnegative integer milliseconds to the nearest car ahead/behind **on track**, independent of race position, class, or lap count. Measured as elapsed time since the leading car crossed the trailing car's current lap position. Excludes pit cars, spectators, pace cars, and cars outside the world. No complete bracket in history means no gap value. |

The tire metadata lookup follows [iRacing's documented index-to-type mapping](https://support.iracing.com/support/solutions/articles/31000176558-2025-season-3-release-notes-2024-06-10-01-).
Fuel, distance, sector times, and gaps are calculated values, not additional native
SDK measurements. The reader requires valid session time, lap fraction, speed,
track length (explicit m/km), pit-road state, and track-surface state for continuity.
Samples separated by more than one second, clock/lap resets, implausible position
jumps, garage/towing, leaving the cockpit, replay, and session/driver/car/track
changes invalidate history. Routine YAML updates preserve history. Gaps retain at
most 180 seconds and 1802 samples per active car (10 Hz history plus the newest
sample). A missing source never turns into a placeholder zero.

## iRacing recorded-file fields

Native `.ibt` files use [IRacingIBTAdapter](../electron/recording/readers/iracing/iracing-ibt-adapter.js)
and an independent channel allowlist. The adapter reuses the common iRacing
conversions and adds 36 mappings: four existing brake-pressure fields, 30 new
shared Physics fields, and two existing coordinate/identity arrays.
Its coverage table accounts for all 271 registered fields,
with 137 supported conditionally on source availability. The live adapter still
maps 101 fields; disk-only values never enter the live capture allowlist.

| Standard fields | Source and units |
| --- | --- |
| `Physics_brake_pressure_{corner}` | `LF/RF/LR/RRbrakeLinePress`, bar, nonnegative. This is hydraulic pressure, separate from normalized driver pedal input. |
| `Physics_abs_cut` | `BrakeABSCutPct`, fraction 0–1 (SDK unit `%`); 0.25 means a 25% reduction in brake force. Separate from the existing ABS activity field and ABS setting. |
| `Physics_tyre_surface_temp_{corner}_{inner/middle/outer}` | `LF/RF/LR/RRtempL/M/R`, degrees Celsius. On left wheels R is inner and L is outer; on right wheels L is inner and R is outer. M is always the middle. |
| `Physics_wheel_speed_{corner}` | `LF/RF/LR/RRspeed`, signed linear tire speed in m/s. No assumed rolling radius or conversion to angular speed/slip. |
| `Physics_ride_height_{corner}` | `LF/RF/LR/RRrideHeight`, meters, signed distance reported at that corner's simulator ride-height reference point. |
| `Physics_suspension_velocity_{corner}` | `LF/RF/LR/RRshockVel`, m/s, signed shock-deflection velocity in the same convention as suspension travel. |
| `Physics_oil_temp` | `OilTemp`, degrees Celsius. |
| `Physics_oil_pressure` | `OilPress`, bar, nonnegative. |
| `Physics_oil_level` | `OilLevel`, liters, nonnegative. |
| `Physics_fuel_pressure` | `FuelPress`, bar, nonnegative. |
| `Physics_manifold_pressure` | `ManifoldPress`, absolute bar, nonnegative; not boost above ambient pressure. |
| `Graphics_car_coordinates` | Player `Lat`/`Lon` (decimal degrees, descriptor `deg`) and `Alt` (meters, descriptor `m`), converted to track-referenced X east / Y up / Z north in meters. Slot 0 holds the player; the other 59 slots are `{x: 0, y: 0, z: 0}` with unavailable IDs. |
| `Graphics_car_id` | Slot 0 holds `PlayerCarIdx`, falling back to session `DriverInfo.DriverCarIdx` when absent. Other slots contain `-1`. `Graphics_player_car_id` holds the same actual player ID; IDs 60–63 are preserved, independent of the 60-slot array limit. |

### Geographic player position

The reference is `WeekendInfo.TrackLatitude`, `TrackLongitude`, and `TrackAltitude`
from the native file. Both reference and car coordinates are converted to
[WGS84 Earth-centered, Earth-fixed XYZ](https://proj.org/en/stable/operations/conversions/cart.html)
using equatorial radius 6378137 m and inverse flattening 298.257223563. Subtracting
the reference and applying the [local tangent-plane rotation](https://proj.org/en/stable/operations/conversions/topocentric.html)
gives east/north/up; the stored XYZ order is east/up/north to preserve Y as height.
Earth-centered coordinates are an intermediate calculation, not the stored values.
Position axes are geographic and fixed to the reference, rather than following
the car's heading. The existing `Physics_heading` and `Physics_velocity_*` fields
retain their simulator-local track basis described above; they are not geographic
heading/velocity components in this position frame.

The origin does not depend on the first car sample, lap, pit visit, or file order.
Files carrying the same track reference therefore align across stints and sessions.
All three reference measurements are required. Native YAML labels track latitude
and longitude with `m` despite containing decimal-degree values (observed in the
saved Lime Rock SDK capture); `deg` and numeric YAML values are also accepted.
This metadata exception does not relax the disk `Lat`/`Lon` descriptor unit check.
Altitude uses the simulator's reported datum for both points; no geoid correction
is available, so these coordinates are not survey-grade absolute elevations.

Invalid/missing coordinates, altitude, units, or player identity omit both arrays.
Coordinates are calculated independently for each sample, with no integration,
interpolation, or carry-forward. Valid zero latitude/longitude/altitude and the
exact origin are retained. Empty array slots are identified by `-1` IDs; no opponent
locations are inferred from lap progress. The map retains an explicitly identified
player at `(0, 0, 0)` while filtering anonymous zero placeholders.

### Disk measurement validation

`{corner}` expands to `front_left`, `front_right`, `rear_left`, or `rear_right`.
The extra mappings require matching units in the file's variable descriptors.
Missing, nonfinite, wrong-unit, wrong-shaped, and out-of-range values are omitted;
valid zero values are retained. Time arrays use their latest sample, following the
existing one-row-per-disk-tick convention. No interpolation creates extra rows.
Explicit off-track/replay flags still suppress physics. Absent live flags default
to cockpit capture only inside this disk adapter. Original files remain unchanged.

Hot `*pressure` channels continue to populate wheel pressure in psi via the common
kPa-to-psi conversion. Surface temperature is never substituted for core
temperature. `*tempC*`, `*wear*`, and `*coldPressure` do not establish continuous
on-track core temperature, wear, or hot pressure and are not substitutes.

Sources: [iRacing's disk brake-pressure and ABS telemetry release notes](https://support.iracing.com/support/solutions/articles/31000170128-2023-season-3-release-notes-2023-06-05-02-),
[AiM's native IBT channel reference](https://www.aim-sportline.com/download/doc/eng/simracing/iRacing_102_eng.pdf),
and [the SDK library's generated channel units](https://irsdk-node.bengsfort.dev/API-Reference/irsdk-node-types/interfaces/TelemetryVarList/).
Tests use synthetic binary IBT files, including unit descriptors, to verify mapping
and worker/JSONL transport. Car-specific availability and physical equivalence with
the ACC reader still require controlled simulator captures; these tests do not
calibrate the legacy ACC brake-pressure signal or its car-specific dash coefficients.

## Contract

This is the application-wide telemetry standard shared by all supported simulators. `Physics_*`, `Graphics_*`, and `Static_*` are application field groups. The application owns this contract, and each simulator reader maps its native data into it before emitting a sample.

- Every game uses this same standard. A game-specific reader maps its SDK/shared-memory fields to these names and types before it emits a sample. A reader omits fields its game cannot supply; it must not create game-prefixed variants, aliases, or additional telemetry names.
- Field meanings, coordinate conventions, units, and enum values are shared across simulators. A reader converts its raw values when necessary to meet the standard before emitting the sample; downstream components do not perform game-specific conversions.
- Standard field names remain stable across readers. A reader uses the semantically equivalent standard field or omits it; it does not introduce a replacement name.
- A successful recorded row is one flat JSON object containing only keys from this catalog, regardless of the source game.
- After a reader has produced the standard object, the writer, saved-file reader, renderer, and upload path preserve every telemetry key and value unchanged. They must not rename fields, add aliases, convert units, wrap the row in another persisted object, or add metadata fields to the row.
- The authoritative field table is [live-telemetry-dataset.js](../src/data/live-telemetry-dataset.js), currently containing 271 keys: 163 Physics, 85 Graphics, and 23 Static. Every reader and adapter must emit rows accepted by this dataset. Register new fields in that table and document their units and meanings here before use; do not introduce game-specific aliases.
- `Graphics_status`, `Graphics_session_type`, `Graphics_flag`, `Graphics_penalty`, `Graphics_track_grip_status`, and the three `Graphics_rain_intensity*` fields contain the standard integers defined below. Readers map native enum values to these integers.
- `Graphics_last_sector_time_str` has type integer despite its suffix, and `Graphics_rain_tyres` is an integer `0`/`1`. Readers must emit these declared types, and downstream components preserve them unchanged.
- `Graphics_car_coordinates` is an array of 60 `{ "x": number, "y": number, "z": number }` objects. `Graphics_car_id` is an array of 60 integers. `Graphics_normalized_positions` is a car-ID-keyed object described below. All other field values are scalar; readers flatten native objects into the exact keys below.
- A cataloged key can be absent from an individual row when its source game cannot supply it or its reader treats its value as unavailable. Absence does not authorize a replacement name.
- Reader-control messages, including `{"available":false}`, remain outside the standard telemetry object and must never be written to the recording or uploaded.

Reader implementations: [ACC reader](../electron/recording/readers/acc/acc-python-reader.js) with its [capture script](../src/py-scripts/ACCMemoryExtractor.py), and [iRacing adapter](../electron/recording/readers/iracing/iracing-adapter.js). Both emit rows validated against the application dataset.

## Physics fields (163)

```text
Physics_abs_cut                                        number
Physics_fuel_pressure                                  number
Physics_manifold_pressure                              number
Physics_oil_level                                      number
Physics_oil_pressure                                   number
Physics_oil_temp                                       number
Physics_ride_height_front_left                          number
Physics_ride_height_front_right                         number
Physics_ride_height_rear_left                           number
Physics_ride_height_rear_right                          number
Physics_suspension_velocity_front_left                  number
Physics_suspension_velocity_front_right                 number
Physics_suspension_velocity_rear_left                   number
Physics_suspension_velocity_rear_right                  number
Physics_tyre_surface_temp_front_left_inner              number
Physics_tyre_surface_temp_front_left_middle             number
Physics_tyre_surface_temp_front_left_outer              number
Physics_tyre_surface_temp_front_right_inner             number
Physics_tyre_surface_temp_front_right_middle            number
Physics_tyre_surface_temp_front_right_outer             number
Physics_tyre_surface_temp_rear_left_inner               number
Physics_tyre_surface_temp_rear_left_middle              number
Physics_tyre_surface_temp_rear_left_outer               number
Physics_tyre_surface_temp_rear_right_inner               number
Physics_tyre_surface_temp_rear_right_middle              number
Physics_tyre_surface_temp_rear_right_outer               number
Physics_wheel_speed_front_left                          number
Physics_wheel_speed_front_right                         number
Physics_wheel_speed_rear_left                           number
Physics_wheel_speed_rear_right                          number
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

## Per-car normalized track positions

`Graphics_normalized_positions` has type `normalized-positions`: a JSON object
mapping canonical nonnegative integer car IDs to finite lap fractions in [0, 1].
Example: `{"0":0,"63":0.75,"1052":1}`. Zero is the start of the lap and one is
the finish. Keys are native simulator car IDs (the same identity domain as
`Graphics_player_car_id` and the values in `Graphics_car_id`), not coordinate
array slots, race positions, or car numbers. There is no 60-car limit. Pit cars
with valid source positions are included. Unavailable cars are omitted, never
assigned zero or -1; an available feed with no valid cars emits `{}`. An
unavailable feed omits the field. The player-only
`Graphics_normalized_car_position` remains a separate scalar.

ACC uses Broadcasting protocol v4 `RealtimeCarUpdate.CarIndex` / `SplinePosition`
alongside shared memory. The [v4 packet layout](https://github.com/EmperorCookie/accapi/blob/main/src/accapi/structs.py)
includes driver count before gear. The client requests 100 ms updates, expires
individual car values after two seconds, and clears them on reconnect, session
or track changes, clock resets, replay transitions, and leaving live capture.
UDP loss never blocks shared-memory sampling. See [ACC setup](../README.md#acc-per-car-track-positions).

iRacing uses each array index in `CarIdxLapDistPct` as the car ID, including
indices 60-63. When `CarIdxTrackSurface` is present, entries outside the world
or without a valid surface are excluded. Missing, negative, nonfinite and
out-of-range fractions are excluded. The map is rebuilt for every live sample
and is also imported from native IBT files when that channel exists. Existing
non-driving/replay gating applies.

## Graphics fields (85)

```text
Graphics_packed_id                                      integer
Graphics_status                                         integer (standard status value)
Graphics_session_type                                   integer (standard session type value)
Graphics_current_time_str                               string
Graphics_last_time_str                                  string
Graphics_best_time_str                                  string
Graphics_last_sector_time_str                           integer
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
Graphics_normalized_positions                           normalized-positions
Graphics_active_cars                                    integer
Graphics_car_coordinates                                array<{x: number, y: number, z: number}>[60]
Graphics_car_id                                         integer[60]
Graphics_player_car_id                                  integer
Graphics_penalty_time                                   number
Graphics_flag                                           integer (standard flag value)
Graphics_penalty                                        integer (standard penalty value)
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
Graphics_track_grip_status                              integer (standard track grip value)
Graphics_rain_intensity                                 integer (standard rain intensity value)
Graphics_rain_intensity_in_10min                        integer (standard rain intensity value)
Graphics_rain_intensity_in_30min                        integer (standard rain intensity value)
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

The numeric meanings below are part of the application contract. Every simulator reader maps its source values to these integers. The descriptions explain the values; readers emit the integers in telemetry rows.

- `Graphics_status`: `0` off, `1` replay, `2` live, `3` paused.
- `Graphics_session_type`: `-1` unknown, `0` practice, `1` qualifying, `2` race, `3` hotlap, `4` time attack, `5` drift, `6` drag, `7` hotstint, `8` hotlap superpole.
- `Graphics_flag`: `0` no flag, `1` blue, `2` yellow, `3` black, `4` white, `5` checkered, `6` penalty, `7` green, `8` orange.
- `Graphics_penalty`: `-1` unknown; `0` no penalty; `1`–`6` cutting penalties; `7`–`12` pit-speeding penalties; `13` disqualified for ignoring a mandatory pit stop; `14` post-race time penalty; `15` disqualified for trolling; `16` disqualified for a pit-entry violation; `17` disqualified for a pit-exit violation; `18` disqualified for driving the wrong way (legacy code); `19` drive-through for ignoring a driver stint; `20` disqualified for ignoring a driver stint; `21` disqualified for exceeding the driver stint limit; `22` disqualified for driving the wrong way.
- `Graphics_track_grip_status`: `0` green, `1` fast, `2` optimum, `3` greasy, `4` damp, `5` wet, `6` flooded.
- Each `Graphics_rain_intensity*` field: `0` no rain, `1` drizzle, `2` light rain, `3` medium rain, `4` heavy rain, `5` thunderstorm.
