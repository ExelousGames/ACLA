# Kestrel — Polished 30-Second Product Explainer

Create a **30-second, silent, 16:9 SaaS product explainer** for **Kestrel**, a sim-racing data analysis assistant.

## Visual direction

Use a premium motorsport-technology aesthetic: dark navy and near-black backgrounds, restrained violet and electric-blue accents, and mint-green success highlights matching the supplied Kestrel UI. Motion should feel precise and purposeful—smooth cursor interaction, controlled camera moves, subtle interface parallax, clean telemetry particles, and elegant match cuts.

Use the supplied screenshots as **exact, undistorted interface plates**. Preserve all original layout, wording, values, colors, proportions, and typography. Do not redraw, regenerate, simplify, crop out essential UI, or invent interface elements. Any glow, outline, cursor, or data trail must sit above the screenshot as a temporary motion-graphics overlay.

## Cursor and motion rules

- Use one small white arrow cursor with a faint violet edge glow.
- Cursor travel uses smooth ease-in/ease-out motion and takes 6–10 frames; never teleport it.
- Each click is shown as: cursor settles for 4 frames, one restrained 110% click-ring pulse, then the resulting action begins.
- Only the controls explicitly named below are clicked. All other emphasis is camera motion or a non-interactive overlay.
- Keep UI plates sharp. Camera scale stays between 100% and 115%; no perspective warping or simulated handheld movement.
- UI-overlay pulses expand once and fade. No repeating glow, flicker, or gaming-style glitch effects.

## Shot-by-shot sequence

### 0:00–0:03.5 — Start baseline collection

**Source plate:** `baseline collection ui- in starting stage.png`

**Framing and movement**

1. Open on the complete screen at 100% scale and hold for 0.35 seconds.
2. Begin a slow digital dolly from 100% to 108%, centered on the baseline collection panel.
3. Add only 2–3 very faint telemetry particles moving left-to-right behind the panel. Nothing passes over text.

**Clicks**

1. Move the cursor to the visible **Full lap** option and click it once at approximately 0:01.2.
2. Move the cursor diagonally to the visible **START / Start collection** control and click it once at approximately 0:02.5.

**Response and transition**

- On the Full lap click, add a brief violet outline around the existing selected state; do not redraw the control.
- On the START click, the button receives one soft violet pulse.
- A violet pulse then travels from the clicked button toward the starting-stage indicator. Follow that pulse into the next plate with a 6-frame match dissolve.

### 0:03.5–0:07 — Recording armed

**Source plate:** `baseline collection ui started.png`

**Framing and movement**

1. Match the previous crop so the panel does not jump during the plate change.
2. Keep the cursor still beside the START control for 0.4 seconds, then fade it out; **no click occurs in this shot**.
3. Pan vertically down the existing stage timeline by approximately 4% of frame height while a single violet telemetry pulse moves down the same path.
4. Add subtle sensor-data dashes drifting toward the panel edges. They must remain outside all labels and values.

**Transition**

- When the pulse reaches the recording/waiting state, let it expand into a thin violet line that wipes downward into the completed-state plate.

### 0:07–0:10 — Baseline ready

**Source plate:** `baseline collection ui finished recording..png`

**Framing and movement**

1. Hold the plate change on the same crop, then ease the camera from 108% back to 103% to reveal more of the completed panel.
2. The existing mint-green checkmark receives one clean 115% completion pulse.
3. A restrained mint glow moves once across **Baseline ready**, then once across **ANALYSIS COMPLETE**.
4. **No cursor and no clicks occur in this shot.**

**Transition**

- Draw 4–5 thin telemetry lines outward from the completed panel. The lines fill the frame and become the abstract analysis visualization.

### 0:10–0:14.5 — Live Performance Analyst

**Visual:** abstract motion-graphics bridge; no screenshot plate.

**Framing and movement**

1. Five differentiated-by-color—not by new on-screen text—telemetry streams represent throttle, brake, steering, speed, and gear.
2. Streams move from small sim-racing sensor nodes on the left into a minimal geometric analysis core at center.
3. The core sorts the streams into aligned traces moving toward the right.
4. Three small mint mistake markers snap onto the outgoing traces one at a time.
5. The camera tracks left-to-right with the data flow. **No cursor and no clicks occur in this shot.**

Do not use a humanoid robot, chatbot bubble, glowing brain, or photoreal racing footage. The graphic should look like an extension of Kestrel’s interface.

**Transition**

- The final mint marker expands into the green result highlight on the next dashboard. Use a clean shape match, not a flash.

### 0:14.5–0:19 — Mistakes identified

**Source plate:** `analysis result showing what mistakes did the driver make.png`

**Framing and movement**

1. Reveal the full dashboard at 100% scale.
2. Push to 110% toward the existing green **13 of 99 total** result.
3. Track laterally across the existing mistake-frequency chart.
4. In sequence, place a thin mint outline around **Hard brake too long**, **Apex too wide**, and **Release brake too slowly**. Each outline holds for approximately 0.35 seconds, then fades before the next appears.
5. **No cursor and no clicks occur in this shot.** Preserve every dashboard value and label exactly.

**Transition**

- Continue the camera’s lateral motion into a gentle downward move, leading naturally to the numbered mistake segments in the same screenshot.

### 0:19–0:24 — Locate each mistake on the lap

**Source plate:** continue using `analysis result showing what mistakes did the driver make.png`

**Framing and movement**

1. Reframe to the numbered mistake-segment area at 108% scale.
2. Move vertically from the first visible numbered segment through segments two, three, four, and five at a constant speed. The screenshot itself remains static; only the camera crop moves.
3. As each segment reaches the center of frame, a precise mint outline traces its existing boundary and its visible mistake tags brighten by no more than 8%.
4. Fade each outline before highlighting the next segment.
5. At approximately 0:23.2, bring the cursor in from the right and stop over the **first visible mistake segment/card**. Do not click yet.

**Transition**

- Hold the hover for 0.25 seconds so the selection is unmistakable.

### 0:24–0:30 — Driver vs Expert

**Source plates:** begin on `analysis result showing what mistakes did the driver make.png`, then use `driver and expert comparison graph.png`

**Click**

1. At approximately 0:24.2, click the hovered **first visible mistake segment/card** once.
2. This is the only click in the final shot.

**Response and movement**

1. A mint selection outline closes around the clicked segment.
2. The outline expands to the comparison-panel bounds while the dashboard dissolves into `driver and expert comparison graph.png` over 8 frames, making the comparison appear to open from the selected segment.
3. Reveal the complete comparison plate at 100% scale; keep throttle, brake, and gear panels sharp and readable.
4. Use narrow overlay masks to reveal the existing green **Driver** path and blue **Expert** path from corner entry to exit. Do not redraw, shift, or change either path.
5. Ease the camera from 106% to 100% so the entire Driver vs Expert view is visible by 0:28.4.
6. Add one subtle mint pulse around the completed comparison, then show the closing title in genuine empty space without covering the UI:

   **BECOME YOUR OWN RACING ANALYST**

7. Lock all movement and hold the final frame from 0:29 to 0:30.

## Edit rhythm and transitions

Every transition follows the product story in one direction: **recording → processing → identifying → locating → comparing**. Use controlled dolly-ins, one deliberate lateral chart move, one vertical segment move, and cursor-led match cuts. Avoid rapid zooms, shaky camera, aggressive montage pacing, and decorative movement unrelated to the user action.

## Audio

**No audio.** Export without narration, music, interface sounds, or a silent audio track.

## Non-negotiable constraints

- No human presenter or talking avatar.
- No photoreal racing footage.
- No generic AI robot, brain, or chatbot imagery.
- No invented interface text, numbers, buttons, charts, controls, states, or logos.
- No screen warping, illegible text, flickering UI, excessive glow, glitching, rapid zooms, or shaky camera.
- Do not imply a click where none is specified.
- Do not animate screenshot contents independently; use camera moves, masks, highlights, and overlays without altering the source plate.
