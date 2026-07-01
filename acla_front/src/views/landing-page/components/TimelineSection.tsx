import React, { useEffect, useRef, useState } from 'react';

/* ── Telemetry Chart (Stage A) ──────────────────── */
const TelemetryChart: React.FC<{ active: boolean }> = ({ active }) => (
  <div className={`lp-tl__chart ${active ? 'lp-tl__chart--active' : ''}`}>
    <svg viewBox="0 0 500 200" preserveAspectRatio="none" className="lp-tl__chart-svg">
      {/* Grid */}
      {[0, 50, 100, 150, 200].map((y) => (
        <line key={y} x1="0" y1={y} x2="500" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      ))}
      {[0, 100, 200, 300, 400, 500].map((x) => (
        <line key={x} x1={x} y1="0" x2={x} y2="200" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      ))}
      {/* Throttle (green) */}
      <polyline
        className="lp-tl__line lp-tl__line--throttle"
        fill="none"
        stroke="var(--lp-green)"
        strokeWidth="2"
        points="0,180 40,170 80,50 120,40 160,40 200,45 240,160 280,170 320,50 360,35 400,40 440,160 480,170 500,180"
      />
      {/* Brake (red) */}
      <polyline
        className="lp-tl__line lp-tl__line--brake"
        fill="none"
        stroke="var(--lp-red)"
        strokeWidth="2"
        points="0,190 40,190 80,190 120,190 160,190 200,180 240,60 280,190 320,190 360,190 400,185 440,50 480,190 500,190"
      />
      {/* Speed (white) */}
      <polyline
        className="lp-tl__line lp-tl__line--speed"
        fill="none"
        stroke="var(--lp-white)"
        strokeWidth="1.5"
        strokeOpacity="0.6"
        points="0,160 40,140 80,80 120,60 160,55 200,65 240,120 280,130 320,70 360,50 400,55 440,110 480,140 500,150"
      />
    </svg>
    {/* Labels */}
    <div className="lp-tl__chart-labels">
      <span style={{ color: 'var(--lp-green)' }}>● Throttle</span>
      <span style={{ color: 'var(--lp-red)' }}>● Brake</span>
      <span style={{ color: 'var(--lp-white)', opacity: 0.6 }}>● Speed</span>
    </div>
  </div>
);

/* ── AI Core (Stage B) ─────────────────────────── */
const AICore: React.FC<{ active: boolean }> = ({ active }) => (
  <div className={`lp-tl__ai ${active ? 'lp-tl__ai--active' : ''}`}>
    <div className="lp-tl__ai-streams" aria-hidden="true">
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="lp-tl__ai-stream"
          style={{
            '--stream-angle': `${i * 45}deg`,
            '--stream-delay': `${i * 0.15}s`,
          } as React.CSSProperties}
        />
      ))}
    </div>
    <div className="lp-tl__ai-core">
      <svg viewBox="0 0 80 80" width="80" height="80">
        <circle cx="40" cy="40" r="30" fill="none" stroke="var(--lp-green)" strokeWidth="1" className="lp-tl__ai-ring lp-tl__ai-ring--outer" />
        <circle cx="40" cy="40" r="20" fill="none" stroke="var(--lp-green)" strokeWidth="1.5" className="lp-tl__ai-ring lp-tl__ai-ring--inner" />
        <circle cx="40" cy="40" r="6" fill="var(--lp-green)" className="lp-tl__ai-dot" />
      </svg>
    </div>
    <div className="lp-tl__ai-output">
      <span className="lp-tl__ai-typewriter">
        "Brake 20m earlier into T4."
      </span>
    </div>
    <div className="lp-tl__ai-samples">
      <span className="lp-tl__ai-sample">"Lift earlier T7"</span>
      <span className="lp-tl__ai-sample">"Box this lap"</span>
      <span className="lp-tl__ai-sample">"Car ahead slow in S2"</span>
    </div>
  </div>
);

/* ── Voice Waveform (Stage C) ──────────────────── */
const VoiceWaveform: React.FC<{ active: boolean }> = ({ active }) => (
  <div className={`lp-tl__voice ${active ? 'lp-tl__voice--active' : ''}`}>
    <div className="lp-tl__voice-bars" aria-hidden="true">
      {Array.from({ length: 32 }).map((_, i) => (
        <div
          key={i}
          className="lp-tl__voice-bar"
          style={{
            '--bar-height': `${20 + Math.sin(i * 0.5) * 40 + Math.random() * 30}%`,
            '--bar-delay': `${i * 0.04}s`,
            '--bar-speed': `${0.4 + Math.random() * 0.4}s`,
          } as React.CSSProperties}
        />
      ))}
    </div>
    <div className="lp-tl__voice-icon">
      <svg viewBox="0 0 48 48" width="48" height="48" fill="none">
        <path
          d="M24 4C20 4 16 8 16 12v10c0 4 4 8 8 8s8-4 8-8V12c0-4-4-8-8-8z"
          stroke="var(--lp-green)"
          strokeWidth="1.5"
        />
        <path d="M10 22c0 8 6 14 14 14s14-6 14-14" stroke="var(--lp-green)" strokeWidth="1.5" strokeLinecap="round" />
        <line x1="24" y1="36" x2="24" y2="44" stroke="var(--lp-green)" strokeWidth="1.5" />
        <line x1="18" y1="44" x2="30" y2="44" stroke="var(--lp-green)" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </div>
  </div>
);

/* ── Stage Icons ───────────────────────────────── */
const StageIcons = {
  gauge: (
    <svg viewBox="0 0 32 32" width="28" height="28" fill="none">
      <circle cx="16" cy="18" r="12" stroke="var(--lp-green)" strokeWidth="1.5" />
      <path d="M16 18l-4-8" stroke="var(--lp-green)" strokeWidth="2" strokeLinecap="round" />
      <circle cx="16" cy="18" r="2" fill="var(--lp-green)" />
    </svg>
  ),
  brain: (
    <svg viewBox="0 0 32 32" width="28" height="28" fill="none">
      <path d="M16 4c-5 0-9 4-9 9 0 3 2 6 4 8l1 2v5h8v-5l1-2c2-2 4-5 4-8 0-5-4-9-9-9z" stroke="var(--lp-green)" strokeWidth="1.5" />
      <path d="M12 28h8" stroke="var(--lp-green)" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M13 14h6M13 18h6" stroke="var(--lp-green)" strokeWidth="1" strokeLinecap="round" opacity="0.5" />
    </svg>
  ),
  wave: (
    <svg viewBox="0 0 32 32" width="28" height="28" fill="none">
      <path d="M4 16c2-4 4-8 6-8s4 16 6 16 4-16 6-16 4 8 6 8" stroke="var(--lp-green)" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
};

/* ── Stage Config ──────────────────────────────── */
const stages = [
  {
    key: 'telemetry',
    title: 'Real-Time Telemetry Intelligence',
    description:
      'Speed, throttle, brake pressure, tire wear, fuel load — monitored continuously and processed in real time.',
    icon: StageIcons.gauge,
    timerRange: [1.0, 0.4] as [number, number],
    Visual: TelemetryChart,
  },
  {
    key: 'ai',
    title: 'Race-Trained AI Engineer',
    description:
      'An AI trained on racing data. It knows when to brake, when to push, and how to overtake — delivered as natural language, not raw numbers.',
    icon: StageIcons.brain,
    timerRange: [0.4, 0.05] as [number, number],
    Visual: AICore,
  },
  {
    key: 'voice',
    title: 'Voice-First, Hands-Free',
    description:
      'AI voiceover reads instructions aloud in real time. No screens to glance at. No buttons to press. Just drive.',
    icon: StageIcons.wave,
    timerRange: [0.05, 0.0] as [number, number],
    Visual: VoiceWaveform,
  },
];

/* ── Main Component ────────────────────────────── */
const TimelineSection: React.FC = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      if (!sectionRef.current) return;
      const rect = sectionRef.current.getBoundingClientRect();
      const sectionH = sectionRef.current.offsetHeight - window.innerHeight;
      if (sectionH <= 0) return;
      const p = Math.max(0, Math.min(1, -rect.top / sectionH));
      setProgress(p);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const activeIndex = progress < 0.33 ? 0 : progress < 0.66 ? 1 : 2;
  const stageProgress = (progress % 0.333) / 0.333;
  const stage = stages[activeIndex];
  const timerVal = stage.timerRange[0] + (stage.timerRange[1] - stage.timerRange[0]) * Math.min(stageProgress, 1);
  const timerDone = progress >= 0.98;

  return (
    <section id="timeline-section" className="lp-tl" ref={sectionRef}>
      <div className="lp-tl__viewport">
        {/* Progress bar */}
        <div className="lp-tl__progress">
          <div className="lp-tl__progress-track">
            <div
              className="lp-tl__progress-fill"
              style={{ height: `${progress * 100}%` }}
            />
            {stages.map((_, i) => (
              <div
                key={i}
                className={`lp-tl__progress-dot ${i <= activeIndex ? 'lp-tl__progress-dot--active' : ''}`}
                style={{ top: `${(i / (stages.length - 1)) * 100}%` }}
              />
            ))}
          </div>
        </div>

        {/* Timer */}
        <div className={`lp-tl__timer ${timerDone ? 'lp-tl__timer--done' : ''}`}>
          {timerVal.toFixed(2)}s
        </div>

        {/* Section title */}
        <h2 className="lp-tl__heading">What Happens in One Second</h2>

        {/* Stages */}
        <div className="lp-tl__stages">
          {stages.map((s, i) => {
            const isActive = i === activeIndex;
            return (
              <div
                key={s.key}
                className={`lp-tl__stage ${isActive ? 'lp-tl__stage--active' : ''}`}
              >
                <div className="lp-tl__stage-visual">
                  <s.Visual active={isActive} />
                </div>
                <div className="lp-tl__stage-card">
                  <div className="lp-tl__stage-icon">{s.icon}</div>
                  <h3 className="lp-tl__stage-title">{s.title}</h3>
                  <p className="lp-tl__stage-desc">{s.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default TimelineSection;
