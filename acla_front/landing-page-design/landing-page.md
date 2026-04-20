# ACLA — Landing Page Structure

---

## Hidden Login

- ACLA logo in the top nav bar acts as a hidden login trigger
- Clicking the logo 5 times rapidly opens a login modal
- No visible "Login" or "Sign In" link on the landing page

---

## Section 0 — Preloader

- SVG racing line (circuit outline) traces across the screen
- ACLA logo appears at center

---

## Section 1 — Hero

Full-screen section. Nav is hidden initially, appears after first scroll.

**Background:** Particle field of telemetry data points streaming left-to-right (green = throttle, red = brake, blue = steering).

**Center content:**

- Headline: **"Your AI Race Engineer. In Real Time."**
- Subline: *"Sub-second analysis. Voice-guided. Hands-free."*
- CTA button: **"See It In Action ↓"** — scrolls to Section 2

**Bottom-right corner:** SVG racing helmet icon

---

## Section 2 — "What Happens in One Second"

Vertical full-page sequence. Each stage occupies one full viewport height. A vertical progress line runs along the left edge, filling downward as the user scrolls through the three stages. A persistent countdown timer (1.00s → 0.00s) ticks down across stages. Each stage has an animated graph on the left and a feature card on the right.

### Stage A — "Real-Time Telemetry Intelligence" (full page)

**Layout:** Left half — animated graph. Right half — feature card.

- **Animated graph:** Live-updating line chart with telemetry channels (throttle, brake, speed) drawing in real time from left to right. Color-coded lines (green, red, white). Grid background with subtle pulse.
- **Card:** Icon: Dashboard gauge. *"Speed, throttle, brake pressure, tire wear, fuel load — monitored continuously and processed in real time."*
- **Timer reads:** 1.00s → 0.40s

### Stage B — "Race-Trained AI Engineer" (full page)

**Layout:** Left half — animated graph. Right half — feature card.

- **Animated graph:** Abstract processing visualization — data streams flow into a glowing core, pulse briefly, then emit a formed sentence via typewriter effect: *"Brake 20m earlier into T4."*
- **Card:** Icon: Brain with circuit-trace patterns. *"An AI trained on racing data. It knows when to brake, when to push, and how to overtake — delivered as natural language, not raw numbers."*
- Sample tips shown on hover: "Lift earlier T7" / "Box this lap" / "Car ahead is slow in S2"
- **Timer reads:** 0.40s → 0.05s

### Stage C — "Voice-First, Hands-Free" (full page)

**Layout:** Left half — animated graph. Right half — feature card.

- **Animated graph:** Audio waveform visualizer — oscillating bars react as if a voice clip is playing. Headset icon with subtle glow animation at center.
- **Card:** Icon: Soundwave / steering wheel. *"AI voiceover reads instructions aloud in real time. No screens to glance at. No buttons to press. Just drive."*
- **Timer reads:** 0.05s → 0.00s — flashes green on completion

---

## Section 3 — "A Lap With ACLA"

Top-down SVG race circuit. A car dot moves along the track as the user scrolls. At 5 key points, a callout annotation appears:

1. **Turn 1 entry:** *"Brake 15m later. You have grip."*
2. **Chicane:** *"Mistake detected: early apex. Lost 0.3s."* (with ⚠ icon)
3. **Back straight:** *"DRS gap closing. Opponent braking 10m early into T6. Overtake window."* (opponent dot visible, gap shrinking)
4. **Hairpin:** *"Tires degrading. Reduce input aggression."* (tire-temp indicator shifts green → orange)
5. **Pit entry:** *"Box now. Undercut window open for 2 laps."* (pit-lane path highlights)

---

## Section 4 — Metrics Bar

Horizontal band with four counters:

- **< 1s** — *"Analysis latency"*
- **200+** — *"Telemetry signals processed per second"*
- **Real-time** — *"Voice delivery"*
- **24/7** — *"Your AI engineer never tires"*

---

## Section 5 — CTA / Closing

Full-screen section.

- Headline: **"Stop Guessing. Start Racing."**
- Subtext: *"Let your AI race engineer handle the thinking."*
- CTA 1: **"Get Early Access"** (primary button)
- CTA 2: **"Watch Demo"** (secondary/outlined button)
