import React, { useEffect, useRef, useState } from 'react';

const TRACK_PATH =
  'M 260 490 L 760 490 C 850 490 900 460 900 400 C 900 355 855 335 815 350 C 770 365 760 320 790 290 C 825 255 880 255 880 200 C 880 150 825 130 775 140 L 470 130 C 375 130 315 165 315 230 C 315 280 375 290 355 335 C 335 380 265 380 225 410 C 195 432 200 470 240 488 Z';

const TURN_MARKERS = [
  { t: 0.085, n: 1 }, { t: 0.16, n: 2 }, { t: 0.24, n: 3 },
  { t: 0.33, n: 4 }, { t: 0.55, n: 5 }, { t: 0.64, n: 6 },
  { t: 0.73, n: 7 }, { t: 0.83, n: 8 }, { t: 0.93, n: 9 },
];

const SECTOR_MARKS = [
  { t: 0.34, label: 'S1' },
  { t: 0.66, label: 'S2' },
];

type AnnType = 'normal' | 'warning' | 'pit' | 'opportunity';

interface Annotation {
  pos: number;
  dx: number;
  dy: number;
  channel: string;
  label: string;
  text: string;
  type: AnnType;
}

const ANNOTATIONS: Annotation[] = [
  { pos: 0.11, dx: 60, dy: -90, channel: 'BRAKE', label: 'T1 ENTRY',
    text: 'Brake 15m later — front tires have grip to give.', type: 'normal' },
  { pos: 0.32, dx: -200, dy: -50, channel: 'LAP DELTA', label: 'CHICANE',
    text: 'Early apex. Lost 0.3s on exit.', type: 'warning' },
  { pos: 0.50, dx: 50, dy: -110, channel: 'STRATEGY', label: 'BACK STRAIGHT',
    text: 'DRS closing on car ahead. Brakes 10m early into T6 — pass window open.', type: 'opportunity' },
  { pos: 0.75, dx: -210, dy: -10, channel: 'TIRES', label: 'HAIRPIN',
    text: 'Front-left graining. Soften steering input through apex.', type: 'warning' },
  { pos: 0.93, dx: -40, dy: 90, channel: 'STRATEGY', label: 'PIT WINDOW',
    text: 'Box this lap. Undercut window open for 2 laps.', type: 'pit' },
];

interface TelemFrame {
  t: number; speed: number; gear: number; throttle: number; brake: number; steer: number;
}

const TELEM_FRAMES: TelemFrame[] = [
  { t: 0.00, speed: 285, gear: 7, throttle: 100, brake: 0,  steer:   0 },
  { t: 0.08, speed: 295, gear: 7, throttle: 100, brake: 0,  steer:   0 },
  { t: 0.11, speed:  95, gear: 2, throttle:   0, brake: 95, steer:  35 },
  { t: 0.17, speed: 145, gear: 3, throttle:  60, brake: 0,  steer:  18 },
  { t: 0.24, speed: 170, gear: 4, throttle:  80, brake: 0,  steer: -22 },
  { t: 0.33, speed: 110, gear: 2, throttle:   0, brake: 80, steer: -38 },
  { t: 0.40, speed: 195, gear: 5, throttle:  90, brake: 0,  steer:   8 },
  { t: 0.50, speed: 285, gear: 7, throttle: 100, brake: 0,  steer:   0 },
  { t: 0.58, speed: 130, gear: 3, throttle:  20, brake: 60, steer: -28 },
  { t: 0.66, speed: 120, gear: 3, throttle:  55, brake: 0,  steer:  30 },
  { t: 0.75, speed:  95, gear: 2, throttle:   0, brake: 85, steer: -36 },
  { t: 0.85, speed: 105, gear: 2, throttle:  20, brake: 50, steer:  28 },
  { t: 0.95, speed: 220, gear: 5, throttle: 100, brake: 0,  steer:   4 },
  { t: 1.00, speed: 280, gear: 7, throttle: 100, brake: 0,  steer:   0 },
];

function sampleTelemetry(t: number): Omit<TelemFrame, 't'> {
  for (let i = 0; i < TELEM_FRAMES.length - 1; i++) {
    const a = TELEM_FRAMES[i], b = TELEM_FRAMES[i + 1];
    if (t >= a.t && t <= b.t) {
      const k = (t - a.t) / Math.max(0.0001, b.t - a.t);
      return {
        speed: Math.round(a.speed + (b.speed - a.speed) * k),
        gear: k < 0.5 ? a.gear : b.gear,
        throttle: Math.round(a.throttle + (b.throttle - a.throttle) * k),
        brake: Math.round(a.brake + (b.brake - a.brake) * k),
        steer: Math.round(a.steer + (b.steer - a.steer) * k),
      };
    }
  }
  const last = TELEM_FRAMES[TELEM_FRAMES.length - 1];
  return { speed: last.speed, gear: last.gear, throttle: last.throttle, brake: last.brake, steer: last.steer };
}

function formatLapTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);
  return `${m}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

interface PathSample { x: number; y: number; angle: number; }

function samplePath(path: SVGPathElement, t: number): PathSample {
  const len = path.getTotalLength();
  const p = path.getPointAtLength(len * Math.max(0, Math.min(1, t)));
  const dt = 0.002;
  const a = path.getPointAtLength(len * Math.max(0, t - dt));
  const b = path.getPointAtLength(len * Math.min(1, t + dt));
  const angle = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI;
  return { x: p.x, y: p.y, angle };
}

interface MarkerState {
  turns: Array<{ t: number; n: number } & PathSample>;
  sectors: Array<{ t: number; label: string } & PathSample>;
  annotations: Array<Annotation & PathSample>;
  start: PathSample | null;
}

const LapSection: React.FC = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const pathRef = useRef<SVGPathElement>(null);
  const [progress, setProgress] = useState(0);
  const [pathLen, setPathLen] = useState(2400);
  const [markers, setMarkers] = useState<MarkerState>({ turns: [], sectors: [], annotations: [], start: null });

  useEffect(() => {
    const handleScroll = () => {
      if (!sectionRef.current) return;
      const rect = sectionRef.current.getBoundingClientRect();
      const h = sectionRef.current.offsetHeight - window.innerHeight;
      if (h <= 0) return;
      const p = Math.max(0, Math.min(1, -rect.top / h));
      setProgress(p);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  /* Snap-scroll: jump to next annotated event on wheel/keyboard/touch.
     Uses a wheel-silence lock so a long trackpad swing = one snap. */
  useEffect(() => {
    const snapPoints = [0, 0.07, ...ANNOTATIONS.map(a => a.pos), 1];
    let locked = false;
    let silenceTimer: ReturnType<typeof setTimeout> | null = null;
    let scrollEndTimer: ReturnType<typeof setTimeout> | null = null;
    const SILENCE_MS = 380;

    const sectionInSticky = () => {
      const section = sectionRef.current;
      if (!section) return null;
      const rect = section.getBoundingClientRect();
      const h = section.offsetHeight - window.innerHeight;
      if (h <= 0) return null;
      if (rect.top > 1 || rect.bottom < window.innerHeight - 1) return null;
      return { rect, h, current: Math.max(0, Math.min(1, -rect.top / h)) };
    };

    const refreshSilence = () => {
      locked = true;
      if (silenceTimer) clearTimeout(silenceTimer);
      silenceTimer = setTimeout(() => { locked = false; }, SILENCE_MS);
    };

    const snapTo = (targetProgress: number, e: Event | null) => {
      const st = sectionInSticky();
      if (!st) return false;
      const sectionTopAbs = window.scrollY + st.rect.top;
      const target = sectionTopAbs + targetProgress * st.h;
      if (e) e.preventDefault();
      window.scrollTo({ top: target, behavior: 'smooth' });
      refreshSilence();
      return true;
    };

    const findNext = (current: number, dir: number): number | undefined => {
      if (dir > 0) return snapPoints.find(p => p > current + 0.02);
      return [...snapPoints].reverse().find(p => p < current - 0.02);
    };

    const findNearest = (current: number) => {
      let best = snapPoints[0], bestD = Math.abs(current - snapPoints[0]);
      for (const p of snapPoints) {
        const d = Math.abs(current - p);
        if (d < bestD) { bestD = d; best = p; }
      }
      return { point: best, distance: bestD };
    };

    const handleWheel = (e: WheelEvent) => {
      const st = sectionInSticky();
      if (!st) return;
      if (locked) {
        e.preventDefault();
        refreshSilence();
        return;
      }
      const dir = e.deltaY > 0 ? 1 : -1;
      const next = findNext(st.current, dir);
      if (next === undefined) return;
      snapTo(next, e);
    };

    const handleKey = (e: KeyboardEvent) => {
      const st = sectionInSticky();
      if (!st) return;
      const downKeys = ['ArrowDown', 'PageDown', ' ', 'Space'];
      const upKeys = ['ArrowUp', 'PageUp'];
      if (downKeys.includes(e.key)) {
        if (locked) { e.preventDefault(); return; }
        const next = findNext(st.current, 1);
        if (next !== undefined) snapTo(next, e);
      } else if (upKeys.includes(e.key)) {
        if (locked) { e.preventDefault(); return; }
        const next = findNext(st.current, -1);
        if (next !== undefined) snapTo(next, e);
      }
    };

    let touchStartY: number | null = null;
    let touchTriggered = false;
    const handleTouchStart = (e: TouchEvent) => {
      const st = sectionInSticky();
      if (!st) return;
      touchStartY = e.touches[0].clientY;
      touchTriggered = false;
    };
    const handleTouchMove = (e: TouchEvent) => {
      const st = sectionInSticky();
      if (!st || touchStartY === null) return;
      e.preventDefault();
      if (touchTriggered || locked) return;
      const dy = touchStartY - e.touches[0].clientY;
      if (Math.abs(dy) < 24) return;
      const dir = dy > 0 ? 1 : -1;
      const next = findNext(st.current, dir);
      if (next === undefined) return;
      touchTriggered = true;
      snapTo(next, e);
    };
    const handleTouchEnd = () => { touchStartY = null; };

    const handleScrollSettle = () => {
      if (scrollEndTimer) clearTimeout(scrollEndTimer);
      scrollEndTimer = setTimeout(() => {
        if (locked) return;
        const st = sectionInSticky();
        if (!st) return;
        const { point, distance } = findNearest(st.current);
        if (distance > 0.008) {
          const sectionTopAbs = window.scrollY + st.rect.top;
          window.scrollTo({ top: sectionTopAbs + point * st.h, behavior: 'smooth' });
          refreshSilence();
        }
      }, 200);
    };

    window.addEventListener('wheel', handleWheel, { passive: false });
    window.addEventListener('keydown', handleKey);
    window.addEventListener('touchstart', handleTouchStart, { passive: false });
    window.addEventListener('touchmove', handleTouchMove, { passive: false });
    window.addEventListener('touchend', handleTouchEnd);
    window.addEventListener('scroll', handleScrollSettle, { passive: true });
    return () => {
      window.removeEventListener('wheel', handleWheel);
      window.removeEventListener('keydown', handleKey);
      window.removeEventListener('touchstart', handleTouchStart);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
      window.removeEventListener('scroll', handleScrollSettle);
      if (silenceTimer) clearTimeout(silenceTimer);
      if (scrollEndTimer) clearTimeout(scrollEndTimer);
    };
  }, []);

  useEffect(() => {
    if (!pathRef.current) return;
    const path = pathRef.current;
    setPathLen(path.getTotalLength());
    setMarkers({
      turns: TURN_MARKERS.map(m => ({ ...m, ...samplePath(path, m.t) })),
      sectors: SECTOR_MARKS.map(m => ({ ...m, ...samplePath(path, m.t) })),
      annotations: ANNOTATIONS.map(a => ({ ...a, ...samplePath(path, a.pos) })),
      start: samplePath(path, 0),
    });
  }, []);

  const car: PathSample = pathRef.current
    ? samplePath(pathRef.current, progress)
    : { x: 260, y: 490, angle: 0 };
  const opp: PathSample = pathRef.current
    ? samplePath(pathRef.current, Math.min(1, progress + 0.07))
    : car;

  /* Cinematic zoom: camera follows the car near annotated events. */
  const FULL_CX = 500, FULL_CY = 280, ZOOM_SCALE = 2.4;
  let zoomIn = 0;
  for (const a of markers.annotations) {
    const d = Math.abs(progress - a.pos);
    const z = Math.max(0, Math.min(1, (0.04 - d) / 0.025));
    const eased = z * z * (3 - 2 * z);
    if (eased > zoomIn) zoomIn = eased;
  }
  const scale = 1 + (ZOOM_SCALE - 1) * zoomIn;
  const tcx = FULL_CX + (car.x - FULL_CX) * zoomIn;
  const tcy = FULL_CY + (car.y - FULL_CY) * zoomIn;
  const groupTransform = `translate(${FULL_CX} ${FULL_CY}) scale(${scale}) translate(${-tcx} ${-tcy})`;
  const project = (x: number, y: number) => ({
    x: (x - tcx) * scale + FULL_CX,
    y: (y - tcy) * scale + FULL_CY,
  });

  const telem = sampleTelemetry(progress);
  const lapTime = formatLapTime(progress * 78.4);
  const delta = (Math.sin(progress * Math.PI * 3.2) * 0.42).toFixed(3);
  const deltaPositive = parseFloat(delta) >= 0;
  const currentSector = progress < 0.34 ? 1 : progress < 0.66 ? 2 : 3;

  /* Title-card → lap transition: intro gone by progress=0.045. */
  const introOpacity = Math.max(0, Math.min(1, 1 - (progress - 0.005) / 0.04));
  const lapOpacity = Math.max(0, Math.min(1, (progress - 0.005) / 0.04));

  return (
    <section className="lp-lap" ref={sectionRef}>
      <div className="lp-lap__viewport">
        <div
          className="lp-lap__intro"
          style={{ opacity: introOpacity, pointerEvents: introOpacity < 0.05 ? 'none' : 'auto' }}
        >
          <div className="lp-lap__intro-chapter">Chapter 03</div>
          <h2 className="lp-lap__intro-title">
            A Lap With <span className="lp-lap__intro-brand">ACLA</span>
          </h2>
          <p className="lp-lap__intro-sub">Suzuka layout · lap 14 / 53</p>
          <div className="lp-lap__intro-hint">Scroll to begin <span>↓</span></div>
        </div>

        <div
          className="lp-lap__layout"
          style={{ opacity: lapOpacity, pointerEvents: lapOpacity < 0.5 ? 'none' : 'auto' }}
        >
          <div className="lp-lap__track-container">
            <svg viewBox="0 0 1000 560" className="lp-lap__svg" overflow="visible">
              <defs>
                <pattern id="checker" x="0" y="0" width="8" height="8" patternUnits="userSpaceOnUse">
                  <rect width="4" height="4" fill="#fff" />
                  <rect x="4" y="4" width="4" height="4" fill="#fff" />
                  <rect x="4" y="0" width="4" height="4" fill="#0a0a0f" />
                  <rect x="0" y="4" width="4" height="4" fill="#0a0a0f" />
                </pattern>
              </defs>

              <g
                transform={groupTransform}
                style={{ transition: 'transform 0.25s cubic-bezier(0.4, 0, 0.2, 1)' }}
              >
                <path
                  d={TRACK_PATH}
                  fill="none"
                  stroke="rgba(255,255,255,0.04)"
                  strokeWidth="48"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                <path
                  d={TRACK_PATH}
                  fill="none"
                  stroke="#1a1a26"
                  strokeWidth="40"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                <path
                  ref={pathRef}
                  d={TRACK_PATH}
                  fill="none"
                  stroke="rgba(255,255,255,0.18)"
                  strokeWidth="1"
                  strokeDasharray="6 5"
                />
                <path
                  d={TRACK_PATH}
                  fill="none"
                  stroke="var(--lp-green)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeDasharray={pathLen}
                  strokeDashoffset={pathLen * (1 - progress)}
                  opacity="0.85"
                  style={{ filter: 'drop-shadow(0 0 4px var(--lp-green-glow))' }}
                />

                {markers.start && (
                  <g transform={`translate(${markers.start.x}, ${markers.start.y}) rotate(${markers.start.angle + 90})`}>
                    <rect x="-22" y="-4" width="44" height="8" fill="url(#checker)" />
                    <text x="0" y="-12" textAnchor="middle" className="lp-lap__sf-text">S/F</text>
                  </g>
                )}

                {markers.sectors.map((s, i) => (
                  <g key={i} transform={`translate(${s.x}, ${s.y}) rotate(${s.angle + 90})`}>
                    <line x1="-24" y1="0" x2="24" y2="0" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
                    <text x="30" y="4" className="lp-lap__sector-text">{s.label}</text>
                  </g>
                ))}

                {markers.turns.map((t, i) => {
                  const rad = ((t.angle - 90) * Math.PI) / 180;
                  const mx = t.x + Math.cos(rad) * 30;
                  const my = t.y + Math.sin(rad) * 30;
                  const passed = progress >= t.t;
                  return (
                    <g key={i} className={`lp-lap__turn ${passed ? 'lp-lap__turn--passed' : ''}`}>
                      <circle
                        cx={mx}
                        cy={my}
                        r="11"
                        fill={passed ? 'var(--lp-green-dim)' : 'rgba(255,255,255,0.04)'}
                        stroke={passed ? 'var(--lp-green)' : 'rgba(255,255,255,0.25)'}
                        strokeWidth="1.5"
                      />
                      <text
                        x={mx}
                        y={my + 4}
                        textAnchor="middle"
                        className="lp-lap__turn-text"
                        style={{ fill: passed ? 'var(--lp-green)' : 'rgba(255,255,255,0.6)' }}
                      >
                        {t.n}
                      </text>
                    </g>
                  );
                })}

                {progress > 0.40 && progress < 0.62 && (
                  <g transform={`translate(${opp.x}, ${opp.y}) rotate(${opp.angle})`}>
                    <circle r="14" fill="var(--lp-red)" opacity="0.12" />
                    <polygon points="-9,-5 -9,5 9,0" fill="var(--lp-red)" opacity="0.85" />
                    <text x="14" y="3" className="lp-lap__opp-label">OPP</text>
                  </g>
                )}

                <g transform={`translate(${car.x}, ${car.y})`}>
                  <circle r="18" fill="var(--lp-green)" opacity="0.12" className="lp-lap__car-glow" />
                  <circle r="11" fill="var(--lp-green)" opacity="0.18" />
                  <g transform={`rotate(${car.angle})`}>
                    <polygon
                      points="-10,-6 -10,6 11,0"
                      fill="var(--lp-green)"
                      style={{ filter: 'drop-shadow(0 0 6px var(--lp-green-glow))' }}
                    />
                    <polygon points="-10,-6 -10,6 -4,0" fill="#0a0a0f" opacity="0.5" />
                  </g>
                </g>
              </g>

              {markers.annotations.map((a, i) => {
                const visible = progress >= a.pos - 0.025 && progress <= a.pos + 0.16;
                const trackP = project(a.x, a.y);
                const cardP = project(a.x + a.dx, a.y + a.dy);
                return (
                  <g key={i} className={`lp-lap__ann ${visible ? 'lp-lap__ann--visible' : ''}`}>
                    <line
                      x1={trackP.x}
                      y1={trackP.y}
                      x2={cardP.x}
                      y2={cardP.y}
                      className={`lp-lap__leader lp-lap__leader--${a.type}`}
                    />
                    <circle
                      cx={trackP.x}
                      cy={trackP.y}
                      r="5"
                      className={`lp-lap__leader-dot lp-lap__leader-dot--${a.type}`}
                    />
                    <foreignObject x={cardP.x} y={cardP.y - 8} width="320" height="170">
                      <div className={`lp-lap__ann-card lp-lap__ann-card--${a.type}`}>
                        <div className="lp-lap__ann-head">
                          <span className="lp-lap__ann-channel">{a.channel}</span>
                          <span className="lp-lap__ann-label">{a.label}</span>
                        </div>
                        <div className="lp-lap__ann-text">&ldquo;{a.text}&rdquo;</div>
                      </div>
                    </foreignObject>
                  </g>
                );
              })}
            </svg>
          </div>

          <aside className="lp-lap__hud">
            <div className="lp-lap__hud-section lp-lap__hud-section--brand">
              <span className="lp-lap__hud-led" />
              <div className="lp-lap__hud-brand-text">
                <div className="lp-lap__hud-title">ACLA</div>
                <div className="lp-lap__hud-live">LIVE</div>
              </div>
            </div>

            <div className="lp-lap__hud-section">
              <div className="lp-lap__hud-label">LAP TIME</div>
              <div className="lp-lap__hud-time-val">{lapTime}</div>
              <div className={`lp-lap__hud-delta ${deltaPositive ? 'lp-lap__hud-delta--pos' : 'lp-lap__hud-delta--neg'}`}>
                Δ {deltaPositive ? '+' : ''}{delta}s
              </div>
            </div>

            <div className="lp-lap__hud-section">
              <div className="lp-lap__hud-label">SECTOR</div>
              <div className="lp-lap__hud-sectors">
                {[1, 2, 3].map(n => (
                  <div
                    key={n}
                    className={`lp-lap__hud-sector ${currentSector === n ? 'lp-lap__hud-sector--active' : ''} ${currentSector > n ? 'lp-lap__hud-sector--done' : ''}`}
                  >
                    S{n}
                  </div>
                ))}
              </div>
            </div>

            <div className="lp-lap__hud-section lp-lap__hud-section--stats">
              <div className="lp-lap__hud-stat">
                <div className="lp-lap__hud-label">SPEED</div>
                <div className="lp-lap__hud-stat-val">
                  {telem.speed}<span>km/h</span>
                </div>
              </div>
              <div className="lp-lap__hud-stat">
                <div className="lp-lap__hud-label">GEAR</div>
                <div className="lp-lap__hud-stat-val lp-lap__hud-stat-val--big">{telem.gear}</div>
              </div>
            </div>

            <div className="lp-lap__hud-section lp-lap__hud-section--channels">
              <HudBar label="THROTTLE" value={telem.throttle} color="var(--lp-green)" />
              <HudBar label="BRAKE" value={telem.brake} color="var(--lp-red)" />
              <HudSteer value={telem.steer} />
            </div>

            <div className="lp-lap__hud-section lp-lap__hud-section--foot">
              <div>
                <div className="lp-lap__hud-label">FUEL</div>
                <div className="lp-lap__hud-foot-val">62%</div>
              </div>
              <div>
                <div className="lp-lap__hud-label">TIRES</div>
                <div className="lp-lap__hud-foot-val">M · 14<span>LAPS</span></div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
};

const HudBar: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div className="lp-lap__hud-channel">
    <div className="lp-lap__hud-channel-head">
      <span className="lp-lap__hud-label">{label}</span>
      <span className="lp-lap__hud-channel-val">{value}</span>
    </div>
    <div className="lp-lap__hud-bar">
      <div
        className="lp-lap__hud-bar-fill"
        style={{ width: `${value}%`, background: color, boxShadow: `0 0 8px ${color}` }}
      />
    </div>
  </div>
);

const HudSteer: React.FC<{ value: number }> = ({ value }) => {
  const pct = ((value + 50) / 100) * 100;
  return (
    <div className="lp-lap__hud-channel">
      <div className="lp-lap__hud-channel-head">
        <span className="lp-lap__hud-label">STEER</span>
        <span className="lp-lap__hud-channel-val">{value > 0 ? '+' : ''}{value}°</span>
      </div>
      <div className="lp-lap__hud-bar lp-lap__hud-bar--center">
        <div className="lp-lap__hud-bar-center" />
        <div className="lp-lap__hud-bar-steer" style={{ left: `${pct}%` }} />
      </div>
    </div>
  );
};

export default LapSection;
