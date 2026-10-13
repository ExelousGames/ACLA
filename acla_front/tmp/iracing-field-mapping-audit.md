# Initial iRacing recording field-mapping audit

Implementation update (2026-09-14): the adapter now declares **100 mapped fields**
and **140 unmapped fields** (Physics 39, Graphics 52, Static 9 mapped). The added
25 mappings cover 15 motion scalars (including derived world velocity), fitted tire compound/wet identity, three fuel
calculations, two sector-time fields, two on-track gaps, and stint distance. See
[the implemented conventions and validity rules](telemetry-fields.md#motion-and-calculated-fields).
Motion transforms still need physical validation with controlled simulator
captures; mapping declarations are not measured live coverage. The audit below is
the **pre-extension snapshot** and retains its original 75/165 inventory.

Audit date: 2026-09-14.

Scope: the live shared-memory recorder before the extension above, compared with the application's 240-field catalog. This is a mapping audit, not a measured capture from a running iRacing session. Native `.ibt` import would be a separate input path; the Python reader opens live IRSDK shared memory.

The adapter declares **75 fields mapped** and **165 unmapped**. “Mapped” means a conversion rule exists, conditional on the source being present, valid, and the driver being in the cockpit; it is not measured live coverage.

| Group | Catalog fields | Currently mapped | Currently unmapped |
| --- | ---: | ---: | ---: |
| Physics | 133 | 24 | 109 |
| Graphics | 84 | 42 | 42 |
| Static | 23 | 9 | 14 |
| Total | 240 | 75 | 165 |

All 165 unmapped fields are accounted for exactly once below:

| Audit classification | Fields | Meaning |
| --- | ---: | --- |
| Live restriction | 8 | Live data cannot currently be treated as the required instantaneous measurement; disk or pit data differs. |
| No verified equivalent | 95 | No exact source/semantic equivalent established. This is not a claim that every future car/build can never expose it. |
| Ambiguous | 50 | Needs a source check, unit/coordinate definition, per-car mapping, or product decision. Several are implementable. |
| Derivable candidate | 12 | Plausible mapping with metadata or stateful calculations, subject to defined reset/validity rules. |

The local contract requires omission when a value is unavailable. Filling every key with 0, false, empty strings, or estimated aliases would misrepresent the recording.

## Evidence and limits

- The authoritative catalog is [live-telemetry-dataset.js](../src/data/live-telemetry-dataset.js); current mappings and the generic unsupported reason are in [iracing-adapter.js](../electron/recording/readers/iracing/iracing-adapter.js).
- [telemetry-fields.md](telemetry-fields.md) requires common units and coordinate conventions, but most fields are only listed with types. Axes, handedness, reference frames, signal scales, and reset semantics need explicit definitions.
- [iracing_sdk.py](../src/py-scripts/iracing_sdk.py) selects only requested variable names. It reads but discards variable descriptions and units, and forwards neither a full variable catalog nor the SDK header version. Adding an adapter rule is insufficient unless its source is also requested.
- The tests use synthetic SDK packets. They establish conversion behavior and transport correctness, not actual availability, timing, or physical equivalence on each iRacing car.
- SDK absence claims here are bounded by the inspected implementation and available documentation. Before implementing car-specific candidates, inspect that car's current runtime variable table and session YAML.
- Official iRacing notes explicitly document [live brake-pressure suppression](https://support.iracing.com/support/solutions/articles/31000170128-2023-season-3-release-notes-2023-06-05-02-), [electric-car fuel units](https://support.iracing.com/support/solutions/articles/31000168028-2022-season-4-release-notes-2022-09-06-02-), and the difference between [average track wetness and permission to use rain tires](https://support.iracing.com/support/solutions/articles/31000172730-2024-season-2-patch-1-release-notes-2024-03-13-01-).
- Existing ACC field meanings can also be inspected in the installed `pyaccsharedmemory.py`; its [upstream source and reproduced SDK field descriptions](https://github.com/rrennoir/PyAccSharedMemory) explain why many of these quantities are simulator-specific.

## Reasons by field group

The final table expands every field name. Group numbers link that inventory to these explanations.

| ID | Classification | Group | Fields | Mapping blocker or required work |
| --- | --- | --- | ---: | --- |
| 1 | Live restriction | Brake pressure | 4 | Live brake-line pressure is deliberately zeroed by iRacing. A present zero cannot be treated as a measured zero. Offline .ibt channels may support a later import after verifying pressure units. [iRacing release notes](https://support.iracing.com/support/solutions/articles/31000170128-2023-season-3-release-notes-2023-06-05-02-) |
| 2 | Live restriction | Tire core temperature | 4 | No verified continuously updated rubber-core equivalent. Pit/garage carcass readings are delayed measurements; inner/middle/outer carcass or surface temperatures also need a definition before reducing them to one core temperature. Do not substitute them. |
| 3 | No verified equivalent | Body and suspension damage | 9 | No verified compatible per-region/per-corner damage scale. Incidents, required/optional repair times, and repair-needed booleans do not measure these quantities. iRacing added repair-needed booleans, not this damage schema. [iRacing release notes](https://support.iracing.com/support/solutions/articles/31000177148-2025-season-4-release-notes-2025-09-08-02-) |
| 4 | No verified equivalent | Tire contact geometry | 36 | No verified exported tire contact point, normal, and heading vectors for all four corners. Chassis pose, suspension deflection, and a track spline cannot recover the exact contact geometry. |
| 5 | No verified equivalent | Brake temperatures and wear | 12 | No verified live channels with the required disc temperatures and remaining pad/disc thickness. A value displayed on the in-car dashboard is not evidence that the SDK exports it. |
| 6 | No verified equivalent | Brake compounds | 2 | No verified common brake-pad compound identifiers. A car-specific setup label, if available, requires a defined identifier mapping. |
| 7 | No verified equivalent | Per-tire slip | 12 | No verified direct equivalent for these twelve per-tire quantities. Vehicle sideslip is not tire slip angle, and a wheel-speed/speed estimate is not the simulator tire-contact result. Exact reconstruction needs unavailable contact velocities, wheel orientation, and rolling-radius data. |
| 8 | No verified equivalent | FFB vibration components | 4 | These are simulator-specific vibration contributions to FFB. Acceleration and ABS state do not reproduce their amplitudes. iRacing rumble-pitch channels describe frequency in Hz, not the target kerb-vibration signal. [iRacing release notes](https://www.iracing.com/2017-season-1-release-notes/) |
| 9 | No verified equivalent | World coordinates for all cars | 1 | No verified live XYZ positions for every car in the required 60-entry array. Player GPS or opponent lap fractions cannot supply exact opponent positions, lateral offsets, and elevation. Track-map projections would be estimates requiring different semantics. |
| 10 | No verified equivalent | Exhaust temperature | 1 | No verified live exhaust-temperature equivalent. Water/oil temperatures are different quantities. |
| 11 | No verified equivalent | Sector yellow flags | 3 | No verified complete yellow state per target sector. A player-local yellow or a session caution bit cannot establish the flags in all three sectors. |
| 12 | No verified equivalent | Rain forecasts | 2 | No verified SDK forecasts for exactly 10 and 30 minutes ahead in the target intensity categories. Current precipitation and weather-mode metadata are not forecasts. |
| 13 | No verified equivalent | Tire set identities | 3 | No verified equivalent to numbered physical tire sets and the selected/strategy set IDs. Compound IDs and counts of tire sets used/available do not identify a reusable physical set. |
| 14 | No verified equivalent | Stints and mandatory pit rules | 6 | No universal equivalent to these ACC-derived rule fields. Driver fair-share requirements, fuel-strategy pit windows, and open/closed pit-lane state have different meanings. Custom event rules plus state tracking might provide a defined subset; do not emit default zero/false. |
| 15 | No verified equivalent | Simulation aid multipliers | 4 | No verified like-for-like multipliers/strengths. Fuel-capacity restrictions, damage enablement, tire-set limits, and stability-related vehicle controls are different settings. |
| 16 | Ambiguous | Driver surname | 1 | The adapter receives a display name, not separately structured given/family names. Splitting the last word is unsafe for multi-part names, suffixes, and naming conventions. Needs structured profile data or a user-provided name. |
| 17 | Ambiguous | Coordinate transforms | 15 | Plausible implementation gap: use the relevant velocity, acceleration, and yaw/pitch/roll channels only after establishing local versus world coordinates, axis ordering, handedness, signs, angle reference/wrapping, gravity treatment, and m/s²-to-g conversion. The current field document does not specify those conventions. |
| 18 | Ambiguous | Wheel angular speed | 4 | Candidate wheel-speed channels must first be verified for current live availability and units. Linear wheel speed in m/s is not angular speed in rad/s; conversion requires a validated effective rolling radius, which is not a universal car constant. |
| 19 | Ambiguous | TC intervention | 1 | TC setting/level or a disabled switch does not prove TC is actively intervening or quantify its action. No verified equivalent to the target TC-in-action signal. |
| 20 | Ambiguous | Boost | 1 | Intake manifold absolute pressure, boost above ambient, and a boost control setting are different. Need target units/meaning, car-specific source, and possibly ambient-pressure subtraction; do not alias a boost switch or map setting. |
| 21 | Ambiguous | Automatic transmission and clutch aid | 2 | No verified live actual-assist state in the reader. Session permission to use an aid does not prove that the driver enabled it. Shifts/clutch motion alone cannot identify the setting. |
| 22 | Ambiguous | Ignition and starter | 2 | Need verified actual ignition/starter state channels for the car. Bound buttons, commands, warning bits, and RPM do not by themselves establish both states. Car-specific channels may make these recoverable. |
| 23 | Ambiguous | Engine running | 1 | RPM > 0 is only a heuristic: the engine can be cranking or rotating while stalled/coasting. Need a verified running-state channel or an explicitly accepted estimate. Electric vehicles also need defined semantics. |
| 24 | Ambiguous | AI control | 1 | An AI-enabled session or an AI opponent does not mean the recorded player car is AI-controlled. Need the actual current controller of the sampled car, including ownership/driver transitions. |
| 25 | Ambiguous | Car ID array | 1 | CarIdx identifiers exist, but the target requires exactly 60 integers aligned with car_coordinates. Define participant selection, ordering, unused slots, and handling of larger SDK arrays/grids; IDs need not be blocked forever merely because coordinates are unavailable. |
| 26 | Ambiguous | In-car engine map | 1 | Engine power maps, fuel-mixture controls, throttle shaping, and hybrid deployment modes are distinct and car-specific. Define which control this field represents for each car. |
| 27 | Ambiguous | Dashboard page indices | 2 | Need verified page-state telemetry and a per-car page mapping. An adjustment command, black-box index, or similarly named control is not automatically a dashboard page ID. |
| 28 | Ambiguous | Lights and wipers | 6 | Verify actual state channels and stage enums per car. A command/button or automatic light policy does not establish the current output state; stage numbers may have different meanings. No generic direct mapping is verified. |
| 29 | Ambiguous | Ideal line and setup UI | 2 | Need actual UI/aid state. A session allowing the racing line does not mean it is displayed, and being in the garage does not necessarily mean the setup menu is visible. |
| 30 | Ambiguous | Lap validity | 1 | No verified direct current-lap-valid flag. Delta-reference validity, incident count, track surface, penalties, and final lap results answer different questions. Any reconstruction needs session rules and full lap history; joining mid-lap must remain unknown. |
| 31 | Ambiguous | Penalty code and time | 2 | The target penalty enum includes specific cutting, pit-speeding, stint and disqualification reasons. SDK flags do not identify all reasons or penalty durations. A slow-down time, tow time, and pit repair time are not interchangeable. Only proven subsets can map. |
| 32 | Ambiguous | Current rain and track state | 3 | iRacing exposes current precipitation and an average TrackWetness estimate, but the target uses categorical rain and grip states, including dry-track rubber/grip classes. Define thresholds and labels; do not copy enum integers or treat dry as optimum. [Rain telemetry](https://support.iracing.com/support/solutions/articles/31000172630-2024-season-2-release-notes-2024-03-05-01-), [wetness telemetry](https://support.iracing.com/support/solutions/articles/31000172730-2024-season-2-patch-1-release-notes-2024-03-13-01-) |
| 33 | Ambiguous | Version metadata | 2 | Clarify whether these legacy names mean simulator build and SDK schema version generically or ACC-specific versions. The capture reader sees the IRSDK header version but does not forward it. Never substitute the hard-coded SDK protocol version for an iRacing simulator build. |
| 34 | Ambiguous | Penalties enabled | 1 | A session flagging mode or incident limit does not summarize all penalty systems with one Boolean. Define the target scope and a verified session-option mapping. |
| 35 | Ambiguous | Online session | 1 | Can likely derive from session/server metadata after defining online: service-connected, multiplayer race server, or any remote session. Distinguish local test, AI, hosted, official, and replay; do not hard-code true. |
| 36 | Derivable candidate | Distance in current stint | 1 | Track distance traveled across samples and reset at defined stint boundaries. LapDist is distance around the current lap, not stint distance. Handle reversing, pit entry, towing, resets, reconnects, and dropped ticks; a mid-stint start lacks earlier distance. |
| 37 | Derivable candidate | Fuel accounting | 3 | Maintain refuel/reset-aware consumption over complete laps/stints, then estimate remaining laps. Define averaging and reject insufficient history. FuelLevel differences alone confuse refuels/resets with consumption. Restrict liter-based outputs to fuel vehicles. |
| 38 | Derivable candidate | Last sector time | 2 | Track sector crossings using SessionTime and sector boundaries, with interpolation/gap handling and reset invalidation. A partial first sector cannot yield a complete time. The local contract deliberately declares BOTH fields as integers despite the _str suffix. |
| 39 | Derivable candidate | Gaps ahead and behind | 2 | Use car timing/lap progress and history after defining on-track relative versus race-order/class gaps, lapping behavior, and milliseconds. CarIdxEstTime/F2Time-like data can help, but copying lap fractions or leaderboard positions is invalid. |
| 40 | Derivable candidate | Tire compound and wet/dry identity | 4 | Resolve the actual fitted tire-compound index against the car/session compound metadata, when exposed. Do not hard-code index 1 as wet, confuse the next pit-service compound with the fitted one, or use WeatherDeclaredWet as proof wet tires are fitted. Multiple dry compounds need a policy for the singular dry name. |

## Existing mappings that also need review

These are additional caveats within the 75 declared mappings, not extra entries in the 165-field inventory. Braced corner lists expand to the four named suffixes.

| Field(s) | Issue | Finding |
| --- | --- | --- |
| `Physics_wheel_pressure_{front_left,front_right,rear_left,rear_right}` | Mapping exists, live source not established | The adapter requests LF/RF/LR/RRpressure. Its own README warns about live-versus-pit pressure. Verify actual live channel availability and freshness; do not claim these four are filled because conversion rules exist. Cold/set pressures are not substitutes. |
| `Physics_fuel; Graphics_mfd_fuel_to_add; Static_max_fuel` | Electric-car and capacity semantics | FuelLevel uses kWh on electric cars. The current reader discards SDK unit descriptions and does not check DriverCarIsElectric before emitting Physics_fuel. Verify pit-service units and capacity/restriction semantics too. [iRacing release notes](https://support.iracing.com/support/solutions/articles/31000168028-2022-season-4-release-notes-2022-09-06-02-) |
| `Physics_abs` | Binary activity versus intervention magnitude | Currently converts BrakeABSactive to numeric 0/1. Appropriate only if the target means binary ABS-in-action. It does not carry brake-force reduction magnitude; BrakeABSCutPct is disk telemetry. [iRacing release notes](https://support.iracing.com/support/solutions/articles/31000170128-2023-season-3-release-notes-2023-06-05-02-) |
| `Physics_final_ff` | Signal normalization | SteeringWheelPctTorqueSign is emitted unchanged. Document the intended normalized range, sign, and treatment of steering stops before claiming equivalent force-feedback amplitudes between simulators. |
| `Physics_suspension_travel_{front_left,front_right,rear_left,rear_right}` | Shock versus wheel travel | Currently maps shockDefl. Verify whether target suspension travel means damper deflection or wheel travel and its reference point. Car-specific motion ratios prevent treating those two quantities as universally identical. |
| `Physics_steer_angle; Physics_clutch` | Input conventions | The steering formula and clutch passthrough are asserted by synthetic fixtures. Validate steering range/sign and clutch released/engaged polarity against the intended shared contract using controlled captures; the field document supplies only numeric types. |
| `Graphics_tc_cut_level` | Car-specific TC2 semantics | Currently maps dcTractionControl2. A second traction-control setting is not necessarily ACC-style TC cut level on every vehicle. |
| `Graphics_number_of_laps` | Finish-only implementation | Currently emits LapCompleted only for SessionState 5/6. This is a deliberate implementation gate, not proof the field is unavailable while driving. Define its relation to Graphics_completed_lap and total race length. |
| `Graphics_status` | Incomplete state coverage | Currently emits only off/replay/live; paused (3) is never produced. IsReplayPlaying and IsOnTrack alone may not cover paused replay or all non-driving states. |
| `Graphics_flag; Graphics_global_yellow` | Lossy flag interpretation | The single flag field chooses one of multiple SDK flags by priority; global_yellow combines local yellow/waving with caution bits. Define whether global means full-course caution or any yellow. The target also has categories without simple SDK matches. |
| `Graphics_active_cars; Static_num_cars` | Participant-count semantics | The first counts every nonnegative CarIdxTrackSurface entry; the second excludes pace cars and spectators. Define which participants each count should include before building aligned 60-slot car arrays. |
| `Static_player_name; Static_player_nick` | Identity semantics | The first stores the entire UserName, and the second stores AbbrevName. These are display/abbreviated names, not necessarily given name and chosen nickname. Static_player_surname remains unmapped. |
| `Graphics_estimated_lap_time; Graphics_estimated_lap_time_str; Graphics_delta_lap_time; Graphics_delta_lap_time_str; Graphics_is_delta_positive` | Reference-lap definition | These are explicitly session-best-relative, with estimated time computed as best plus valid delta. They are valid implementation choices only if the shared field meaning uses that reference; an optimal/personal-best prediction would differ. |
| `Static_max_rpm; Static_max_fuel` | Maximum definition | Clarify redline versus hard limiter and nominal tank capacity versus event-restricted usable capacity. The current mappings use DriverCarRedLine and DriverCarFuelMaxLtr. |

## Exact inventory of all 165 currently unmapped fields

| Field | Classification | Reason group |
| --- | --- | --- |
| `Physics_abs_vibration` | No verified equivalent | 8. FFB vibration components |
| `Physics_autoshifter_on` | Ambiguous | 21. Automatic transmission and clutch aid |
| `Physics_brake_pressure_front_left` | Live restriction | 1. Brake pressure |
| `Physics_brake_pressure_front_right` | Live restriction | 1. Brake pressure |
| `Physics_brake_pressure_rear_left` | Live restriction | 1. Brake pressure |
| `Physics_brake_pressure_rear_right` | Live restriction | 1. Brake pressure |
| `Physics_brake_temp_front_left` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_brake_temp_front_right` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_brake_temp_rear_left` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_brake_temp_rear_right` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_car_damage_center` | No verified equivalent | 3. Body and suspension damage |
| `Physics_car_damage_front` | No verified equivalent | 3. Body and suspension damage |
| `Physics_car_damage_left` | No verified equivalent | 3. Body and suspension damage |
| `Physics_car_damage_rear` | No verified equivalent | 3. Body and suspension damage |
| `Physics_car_damage_right` | No verified equivalent | 3. Body and suspension damage |
| `Physics_disc_life_front_left` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_disc_life_front_right` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_disc_life_rear_left` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_disc_life_rear_right` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_front_brake_compound` | No verified equivalent | 6. Brake compounds |
| `Physics_g_force_x` | Ambiguous | 17. Coordinate transforms |
| `Physics_g_force_y` | Ambiguous | 17. Coordinate transforms |
| `Physics_g_force_z` | Ambiguous | 17. Coordinate transforms |
| `Physics_g_vibration` | No verified equivalent | 8. FFB vibration components |
| `Physics_heading` | Ambiguous | 17. Coordinate transforms |
| `Physics_ignition_on` | Ambiguous | 22. Ignition and starter |
| `Physics_is_ai_controlled` | Ambiguous | 24. AI control |
| `Physics_is_engine_running` | Ambiguous | 23. Engine running |
| `Physics_kerb_vibration` | No verified equivalent | 8. FFB vibration components |
| `Physics_local_angular_vel_x` | Ambiguous | 17. Coordinate transforms |
| `Physics_local_angular_vel_y` | Ambiguous | 17. Coordinate transforms |
| `Physics_local_angular_vel_z` | Ambiguous | 17. Coordinate transforms |
| `Physics_local_velocity_x` | Ambiguous | 17. Coordinate transforms |
| `Physics_local_velocity_y` | Ambiguous | 17. Coordinate transforms |
| `Physics_local_velocity_z` | Ambiguous | 17. Coordinate transforms |
| `Physics_pad_life_front_left` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_pad_life_front_right` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_pad_life_rear_left` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_pad_life_rear_right` | No verified equivalent | 5. Brake temperatures and wear |
| `Physics_pitch` | Ambiguous | 17. Coordinate transforms |
| `Physics_rear_brake_compound` | No verified equivalent | 6. Brake compounds |
| `Physics_roll` | Ambiguous | 17. Coordinate transforms |
| `Physics_slip_angle_front_left` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_angle_front_right` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_angle_rear_left` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_angle_rear_right` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_ratio_front_left` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_ratio_front_right` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_ratio_rear_left` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_ratio_rear_right` | No verified equivalent | 7. Per-tire slip |
| `Physics_slip_vibration` | No verified equivalent | 8. FFB vibration components |
| `Physics_starter_engine_on` | Ambiguous | 22. Ignition and starter |
| `Physics_suspension_damage_front_left` | No verified equivalent | 3. Body and suspension damage |
| `Physics_suspension_damage_front_right` | No verified equivalent | 3. Body and suspension damage |
| `Physics_suspension_damage_rear_left` | No verified equivalent | 3. Body and suspension damage |
| `Physics_suspension_damage_rear_right` | No verified equivalent | 3. Body and suspension damage |
| `Physics_tc` | Ambiguous | 19. TC intervention |
| `Physics_turbo_boost` | Ambiguous | 20. Boost |
| `Physics_tyre_contact_heading_front_left_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_front_left_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_front_left_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_front_right_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_front_right_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_front_right_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_rear_left_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_rear_left_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_rear_left_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_rear_right_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_rear_right_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_heading_rear_right_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_front_left_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_front_left_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_front_left_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_front_right_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_front_right_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_front_right_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_rear_left_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_rear_left_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_rear_left_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_rear_right_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_rear_right_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_normal_rear_right_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_front_left_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_front_left_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_front_left_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_front_right_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_front_right_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_front_right_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_rear_left_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_rear_left_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_rear_left_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_rear_right_x` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_rear_right_y` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_contact_point_rear_right_z` | No verified equivalent | 4. Tire contact geometry |
| `Physics_tyre_core_temp_front_left` | Live restriction | 2. Tire core temperature |
| `Physics_tyre_core_temp_front_right` | Live restriction | 2. Tire core temperature |
| `Physics_tyre_core_temp_rear_left` | Live restriction | 2. Tire core temperature |
| `Physics_tyre_core_temp_rear_right` | Live restriction | 2. Tire core temperature |
| `Physics_velocity_x` | Ambiguous | 17. Coordinate transforms |
| `Physics_velocity_y` | Ambiguous | 17. Coordinate transforms |
| `Physics_velocity_z` | Ambiguous | 17. Coordinate transforms |
| `Physics_wheel_angular_s_front_left` | Ambiguous | 18. Wheel angular speed |
| `Physics_wheel_angular_s_front_right` | Ambiguous | 18. Wheel angular speed |
| `Physics_wheel_angular_s_rear_left` | Ambiguous | 18. Wheel angular speed |
| `Physics_wheel_angular_s_rear_right` | Ambiguous | 18. Wheel angular speed |
| `Physics_wheel_slip_front_left` | No verified equivalent | 7. Per-tire slip |
| `Physics_wheel_slip_front_right` | No verified equivalent | 7. Per-tire slip |
| `Physics_wheel_slip_rear_left` | No verified equivalent | 7. Per-tire slip |
| `Physics_wheel_slip_rear_right` | No verified equivalent | 7. Per-tire slip |
| `Graphics_car_coordinates` | No verified equivalent | 9. World coordinates for all cars |
| `Graphics_car_id` | Ambiguous | 25. Car ID array |
| `Graphics_current_tyre_set` | No verified equivalent | 13. Tire set identities |
| `Graphics_direction_light_left` | Ambiguous | 28. Lights and wipers |
| `Graphics_direction_light_right` | Ambiguous | 28. Lights and wipers |
| `Graphics_distance_traveled` | Derivable candidate | 36. Distance in current stint |
| `Graphics_driver_stint_time_left` | No verified equivalent | 14. Stints and mandatory pit rules |
| `Graphics_driver_stint_total_time_left` | No verified equivalent | 14. Stints and mandatory pit rules |
| `Graphics_engine_map` | Ambiguous | 26. In-car engine map |
| `Graphics_exhaust_temp` | No verified equivalent | 10. Exhaust temperature |
| `Graphics_flashing_light` | Ambiguous | 28. Lights and wipers |
| `Graphics_fuel_estimated_laps` | Derivable candidate | 37. Fuel accounting |
| `Graphics_fuel_per_lap` | Derivable candidate | 37. Fuel accounting |
| `Graphics_gap_ahead` | Derivable candidate | 39. Gaps ahead and behind |
| `Graphics_gap_behind` | Derivable candidate | 39. Gaps ahead and behind |
| `Graphics_global_yellow_s1` | No verified equivalent | 11. Sector yellow flags |
| `Graphics_global_yellow_s2` | No verified equivalent | 11. Sector yellow flags |
| `Graphics_global_yellow_s3` | No verified equivalent | 11. Sector yellow flags |
| `Graphics_ideal_line_on` | Ambiguous | 29. Ideal line and setup UI |
| `Graphics_is_setup_menu_visible` | Ambiguous | 29. Ideal line and setup UI |
| `Graphics_is_valid_lap` | Ambiguous | 30. Lap validity |
| `Graphics_last_sector_time` | Derivable candidate | 38. Last sector time |
| `Graphics_last_sector_time_str` | Derivable candidate | 38. Last sector time |
| `Graphics_light_stage` | Ambiguous | 28. Lights and wipers |
| `Graphics_main_display_index` | Ambiguous | 27. Dashboard page indices |
| `Graphics_mandatory_pit_done` | No verified equivalent | 14. Stints and mandatory pit rules |
| `Graphics_mfd_tyre_set` | No verified equivalent | 13. Tire set identities |
| `Graphics_missing_mandatory_pits` | No verified equivalent | 14. Stints and mandatory pit rules |
| `Graphics_penalty` | Ambiguous | 31. Penalty code and time |
| `Graphics_penalty_time` | Ambiguous | 31. Penalty code and time |
| `Graphics_rain_intensity` | Ambiguous | 32. Current rain and track state |
| `Graphics_rain_intensity_in_10min` | No verified equivalent | 12. Rain forecasts |
| `Graphics_rain_intensity_in_30min` | No verified equivalent | 12. Rain forecasts |
| `Graphics_rain_light` | Ambiguous | 28. Lights and wipers |
| `Graphics_rain_tyres` | Derivable candidate | 40. Tire compound and wet/dry identity |
| `Graphics_secondary_display_index` | Ambiguous | 27. Dashboard page indices |
| `Graphics_strategy_tyre_set` | No verified equivalent | 13. Tire set identities |
| `Graphics_track_grip_status` | Ambiguous | 32. Current rain and track state |
| `Graphics_track_status` | Ambiguous | 32. Current rain and track state |
| `Graphics_tyre_compound` | Derivable candidate | 40. Tire compound and wet/dry identity |
| `Graphics_used_fuel` | Derivable candidate | 37. Fuel accounting |
| `Graphics_wiper_stage` | Ambiguous | 28. Lights and wipers |
| `Static_ac_version` | Ambiguous | 33. Version metadata |
| `Static_aid_auto_clutch` | Ambiguous | 21. Automatic transmission and clutch aid |
| `Static_aid_fuel_rate` | No verified equivalent | 15. Simulation aid multipliers |
| `Static_aid_mechanical_damage` | No verified equivalent | 15. Simulation aid multipliers |
| `Static_aid_stability` | No verified equivalent | 15. Simulation aid multipliers |
| `Static_aid_tyre_rate` | No verified equivalent | 15. Simulation aid multipliers |
| `Static_dry_tyres_name` | Derivable candidate | 40. Tire compound and wet/dry identity |
| `Static_is_online` | Ambiguous | 35. Online session |
| `Static_penalty_enabled` | Ambiguous | 34. Penalties enabled |
| `Static_pit_window_end` | No verified equivalent | 14. Stints and mandatory pit rules |
| `Static_pit_window_start` | No verified equivalent | 14. Stints and mandatory pit rules |
| `Static_player_surname` | Ambiguous | 16. Driver surname |
| `Static_sm_version` | Ambiguous | 33. Version metadata |
| `Static_wet_tyres_name` | Derivable candidate | 40. Tire compound and wet/dry identity |

## Practical next work

1. Define the missing cross-simulator units, coordinate frames, signal scales, and state/reset rules.
2. Inspect an actual current iRacing variable table, including descriptions/units, plus session YAML for the intended cars. Validate transitions such as steering, braking, pit entry/refuel, reset, and driver change.
3. Prioritize motion-vector conversion, complete-sector timing, fuel accounting, tire-compound lookup, and timing gaps. These provide useful coverage without inventing measurements.
4. Treat native `.ibt` import as a separate capability for data deliberately withheld from live telemetry. It can recover some measurements, including brake pressure, but does not solve every semantic mismatch or supply exact opponent XYZ positions.
5. Replace the generic unsupported reason with field-specific capability reasons after implementation decisions are verified. Keep unavailable fields absent under the existing contract.
