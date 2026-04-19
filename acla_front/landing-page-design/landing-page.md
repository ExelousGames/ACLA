# ACLA — Real-Time Sim Racing AI Agent: Landing Page Design

---

## Section 0 — Preloader

A minimal dark screen. A thin SVG racing line traces itself from left to right in neon cyan, mimicking a circuit outline. The stroke animates with `stroke-dashoffset`, finishing as the page loads. The ACLA logo fades in at center — glitch-style letter reveal, each character offset by 40ms.

---

## Section 1 — Hero (Viewport-locked, Full Bleed)

**Layout:** Full-screen dark canvas (#0A0A0F). No nav visible initially — it fades in after the first scroll trigger.

**Background:** A looping WebGL particle field simulating telemetry data points streaming left-to-right at variable speeds. Particles are color-coded: green (throttle), red (brake), blue (steering angle) — creating a living data river. Parallax depth: particles in the foreground move 2× scroll speed, background particles at 0.5×.

**Center content:**

- Headline: **"Your AI Race Engineer. In Real Time."** — Each word staggers in from below with spring easing (stiffness: 120, damping: 14). Typeface: geometric sans, ultra-bold, ~96px.
- Subline fades in 300ms after: *"Sub-second analysis. Voice-guided. Hands-free."* — lighter weight, muted white (#B0B0C0).
- A single CTA button: **"See It In Action ↓"** — pill-shaped, outlined in cyan. On hover: fills with a gradient sweep left-to-right (micro-interaction). On click: smooth-scrolls to Section 2.

**Micro-interaction:** A subtle SVG racing helmet icon in the bottom-right corner breathes (scale 1.0 → 1.03 → 1.0 on a 3s loop). Mouse cursor proximity within 200px causes the helmet visor to "glow" — a radial gradient opacity shift.

---

## Section 2 — Scrollytelling: "What Happens in One Second"

**Mechanic:** Scroll-pinned. The viewport locks as the user scrolls through ~4 scroll-lengths of content. A central timeline bar (horizontal, SVG) fills left-to-right as the user scrolls, representing **1 second of real time**.

**Stage A (0–25% scroll):** Label appears: *"Telemetry Captured"*. An SVG car silhouette appears at left. Data lines (throttle, brake, speed, tire temp) animate outward from the car like an ECG, drawn with `stroke-dasharray` animation. Parallax: the car stays fixed, data lines scroll slightly faster.

**Stage B (25–50% scroll):** Label: *"ML Models Analyze"*. The data lines converge into a central node — a pulsing neural-network SVG. Nodes light up sequentially (micro-interaction: each node scales up 10% then settles). Connection lines between nodes glow in sequence, implying inference.

**Stage C (50–75% scroll):** Label: *"LLM Understands Context"*. The neural network morphs (SVG path interpolation) into a speech-bubble shape. Inside, text types itself out character-by-character: *"Brake 20m earlier into T4. You're overdriving entry."* — monospace font, terminal-green color.

**Stage D (75–100% scroll):** Label: *"Voice Delivers. Hands Free."*. A soundwave SVG animates — bars oscillating at varied heights. The speech bubble shrinks and a stylized ear/headset icon fades in. A faint audio waveform ripple expands outward (CSS radial keyframe). Text: *"All in under one second."* — bold, centered, fade-in.

---

## Section 3 — Feature Triptych (Parallax Cards)

**Layout:** Three tall cards arranged horizontally on desktop (stacked on mobile). Each card is a dark glass-morphism panel (background blur, 1px white border at 8% opacity). As the user scrolls, the cards rise into view at staggered intervals (card 1 at scroll 0%, card 2 at 15%, card 3 at 30%) with `translateY(60px) → 0` and opacity 0 → 1.

### Card 1 — "Real-Time Telemetry Intelligence"

- Icon: Animated SVG dashboard gauge. The needle sweeps from 0 to redline on scroll-enter, then settles with a spring bounce.
- Body: *"Speed, throttle, brake pressure, tire wear, fuel load — monitored continuously by an ensemble of ML models tuned for racing physics."*
- Micro-interaction: Hovering the card triggers faint grid lines to appear behind the gauge, simulating a data overlay.

### Card 2 — "Race-Trained LLM Engineer"

- Icon: SVG brain with circuit-trace patterns. On scroll-enter, the traces illuminate sequentially like electricity flowing through pathways.
- Body: *"A purpose-built language model trained on racing data. It knows when to brake, when to push, what mistakes you made, how to overtake, and when to pit — delivered as natural language, not raw numbers."*
- Micro-interaction: Hover causes a small chat-bubble to peek out from the bottom-right corner of the card with rotating sample tips: "Lift earlier T7" → "Box this lap" → "Car ahead is slow in S2".

### Card 3 — "Voice-First, Hands-Free"

- Icon: SVG soundwave morphing into a steering wheel outline (path interpolation on scroll). Conveys: audio + driving = unified.
- Body: *"AI voiceover reads instructions aloud in real time. No screens to glance at. No buttons to press. Just drive."*
- Micro-interaction: Hover plays a CSS-only equalizer animation (5 bars oscillating at different frequencies).

---

## Section 4 — Scrollytelling: "A Lap With ACLA"

**Mechanic:** A top-down SVG race circuit (vector, clean, geometric — not photorealistic). A dot representing the car moves along the circuit path as the user scrolls. The path uses `getPointAtLength()` driven by scroll position.

At **5 key points** along the circuit, the car dot pauses and a callout annotation pops in (scale 0 → 1, spring ease):

1. **Turn 1 entry:** *"Brake 15m later. You have grip."* — annotation in green.
2. **Chicane:** *"Mistake detected: early apex. Lost 0.3s."* — annotation in amber with a small ⚠ SVG icon pulse.
3. **Back straight:** *"DRS gap closing. Opponent braking 10m early into T6. Overtake window."* — annotation in cyan. A second dot (opponent) appears and the gap shrinks visually.
4. **Hairpin:** *"Tires degrading. Reduce input aggression."* — annotation in orange. The tire-temp SVG micro-graphic next to the callout shifts from green to orange gradient.
5. **Pit entry:** *"Box now. Undercut window open for 2 laps."* — annotation in white. A pit-lane branch of the SVG path highlights.

**Parallax:** The circuit background layer scrolls at 0.7× speed. Callout annotations at 1×. Faint telemetry sparklines float in the deep background at 0.3×.

---

## Section 5 — Tech Stack Visualization (Motion UI)

**Layout:** A centered, radial diagram. The ACLA logo sits at the center. Orbiting around it in concentric rings are labeled nodes:

- **Inner ring (ML Models):** Nodes labeled "Tire Model", "Braking Classifier", "Overtake Predictor", "Fuel Strategy Net" — each node is a small SVG hexagon. They orbit slowly (CSS `@keyframes rotate`, 60s per revolution). On hover, a node stops and expands, revealing a one-line description.

- **Outer ring (Data Sources):** "Throttle", "Brake", "Steering", "GPS", "Tire Temp", "Fuel Level" — circular nodes connected to inner ring by faint dashed lines (SVG, animated dash offset for a "data flowing" effect).

**Scroll trigger:** The entire diagram starts collapsed (all nodes at center) and expands outward as the user scrolls into view — a big-bang animation over 800ms, staggered by ring.

---

## Section 6 — Social Proof / Metrics Bar

**Layout:** A single horizontal band, dark background with a subtle noise texture. Four counters:

- **< 1s** — *"Analysis latency"*
- **200+** — *"Telemetry signals processed per second"*
- **Real-time** — *"Voice delivery"*
- **24/7** — *"Your AI engineer never tires"*

**Motion:** Numbers count up from 0 when scrolled into view (CountUp.js style). Each counter is separated by a thin vertical SVG line that draws itself top-to-bottom on enter.

---

## Section 7 — CTA / Closing

**Layout:** Full viewport height. Deep black fading to a subtle dark blue-purple gradient at the bottom edge.

**Content:**

- Large headline: **"Stop Guessing. Start Racing."** — words reveal one by one, each with a downward mask-wipe animation.
- Subtext: *"Let your AI race engineer handle the thinking."*
- Two CTAs side by side: **"Get Early Access"** (filled cyan button, hover: slight lift + shadow) and **"Watch Demo"** (outlined button, hover: border color shift cyan → white).

**Background motion:** Faint, large-scale SVG checkered flag pattern slowly waves (CSS transform: skewX oscillation on a loop) at 5% opacity — barely visible, subliminally reinforcing the racing context.

---

## Global Design Notes

| Aspect | Spec |
|---|---|
| **Color palette** | Near-black base (#0A0A0F), cyan accent (#00F0FF), alert red (#FF3B4A), data green (#00E676), warm amber (#FFB300), muted text (#B0B0C0) |
| **Typography** | Headings: geometric sans (e.g., Space Grotesk or Outfit), bold/black. Body: Inter or similar, regular weight. Monospace accents for data/code-like text |
| **Scroll library** | GSAP ScrollTrigger for pinning and timeline scrubbing. Lenis or Locomotive for smooth scroll normalization |
| **SVG animation** | Inline SVGs animated via GSAP or CSS `stroke-dashoffset`, `path` morphing via flubber or GSAP MorphSVG |
| **Parallax** | Layered `data-speed` attributes on elements, driven by scroll position via GSAP or a lightweight parallax lib |
| **Micro-interactions** | Hover states on every interactive element. Cursor-proximity reactions on hero elements. Spring physics for enters/exits (Framer Motion or GSAP) |
| **Performance** | `will-change` on animated layers. `IntersectionObserver` to activate/deactivate off-screen animations. Lazy-load below-fold SVGs. Target 60fps |
| **Accessibility** | `prefers-reduced-motion` media query disables parallax, pins, and looping animations — content remains fully readable in static layout. All SVGs carry `role="img"` + `aria-label` |
