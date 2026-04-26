import React, { useEffect, useRef, useState } from 'react';

/* Track path (simplified circuit) */
const TRACK_PATH =
  'M 300,70 L 520,55 Q 620,50 660,110 L 690,200 Q 710,260 670,310 L 580,380 Q 540,410 480,415 L 320,425 Q 260,425 220,400 L 140,350 Q 100,320 85,270 L 70,190 Q 60,130 100,100 L 190,75 Q 230,65 300,70 Z';

/* Annotations at specific path positions (0-1) */
const ANNOTATIONS = [
  {
    position: 0.08,
    label: 'T1',
    text: '"Brake 15m later. You have grip."',
    side: 'right' as const,
    type: 'normal' as const,
  },
  {
    position: 0.28,
    label: 'Chicane',
    text: '"Mistake detected: early apex. Lost 0.3s."',
    side: 'left' as const,
    type: 'warning' as const,
  },
  {
    position: 0.48,
    label: 'Straight',
    text: '"DRS gap closing. Opponent braking 10m early into T6. Overtake window."',
    side: 'right' as const,
    type: 'normal' as const,
  },
  {
    position: 0.7,
    label: 'Hairpin',
    text: '"Tires degrading. Reduce input aggression."',
    side: 'left' as const,
    type: 'warning' as const,
  },
  {
    position: 0.9,
    label: 'Pit',
    text: '"Box now. Undercut window open for 2 laps."',
    side: 'right' as const,
    type: 'pit' as const,
  },
];

const LapSection: React.FC = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const pathRef = useRef<SVGPathElement>(null);
  const [progress, setProgress] = useState(0);
  const [carPos, setCarPos] = useState({ x: 300, y: 70 });
  const [opponentPos, setOpponentPos] = useState({ x: 300, y: 70 });

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
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (!pathRef.current) return;
    const len = pathRef.current.getTotalLength();
    const pt = pathRef.current.getPointAtLength(len * progress);
    setCarPos({ x: pt.x, y: pt.y });
    // Opponent is slightly ahead
    const oppProgress = Math.min(1, progress + 0.06);
    const oppPt = pathRef.current.getPointAtLength(len * oppProgress);
    setOpponentPos({ x: oppPt.x, y: oppPt.y });
  }, [progress]);

  return (
    <section className="lp-lap" ref={sectionRef}>
      <div className="lp-lap__viewport">
        <h2 className="lp-lap__heading">A Lap With ACLA</h2>

        <div className="lp-lap__track-container">
          <svg viewBox="0 0 780 480" className="lp-lap__svg">
            {/* Track surface */}
            <path
              d={TRACK_PATH}
              fill="none"
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="36"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {/* Track center line */}
            <path
              ref={pathRef}
              d={TRACK_PATH}
              fill="none"
              stroke="rgba(255,255,255,0.15)"
              strokeWidth="1"
              strokeDasharray="6 4"
            />
            {/* Driven portion */}
            <path
              d={TRACK_PATH}
              fill="none"
              stroke="var(--lp-green)"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeDasharray={pathRef.current?.getTotalLength() || 2000}
              strokeDashoffset={
                (pathRef.current?.getTotalLength() || 2000) * (1 - progress)
              }
              opacity="0.6"
            />
            {/* Pit lane hint (appears late) */}
            {progress > 0.82 && (
              <path
                d="M 200,400 Q 160,440 130,440 L 80,440 Q 50,440 50,410 L 50,300"
                fill="none"
                stroke="var(--lp-blue)"
                strokeWidth="2"
                strokeDasharray="4 3"
                opacity="0.5"
                className="lp-lap__pit-lane"
              />
            )}
            {/* Opponent dot */}
            {progress > 0.35 && progress < 0.65 && (
              <g>
                <circle cx={opponentPos.x} cy={opponentPos.y} r="8" fill="var(--lp-red)" opacity="0.25" />
                <circle cx={opponentPos.x} cy={opponentPos.y} r="4" fill="var(--lp-red)" opacity="0.7" />
              </g>
            )}
            {/* Car dot */}
            <circle cx={carPos.x} cy={carPos.y} r="10" fill="var(--lp-green)" opacity="0.2" className="lp-lap__car-glow" />
            <circle cx={carPos.x} cy={carPos.y} r="5" fill="var(--lp-green)" className="lp-lap__car" />
          </svg>

          {/* Annotations */}
          {ANNOTATIONS.map((ann, i) => {
            const visible = progress >= ann.position - 0.02 && progress <= ann.position + 0.15;
            return (
              <div
                key={i}
                className={`lp-lap__annotation lp-lap__annotation--${ann.side} lp-lap__annotation--${ann.type} ${visible ? 'lp-lap__annotation--visible' : ''}`}
                style={{ '--ann-index': i } as React.CSSProperties}
              >
                <span className="lp-lap__annotation-label">
                  {ann.type === 'warning' && '⚠ '}
                  {ann.type === 'pit' && '🏁 '}
                  {ann.label}
                </span>
                <span className="lp-lap__annotation-text">{ann.text}</span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default LapSection;
