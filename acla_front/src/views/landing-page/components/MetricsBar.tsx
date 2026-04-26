import React, { useEffect, useRef, useState } from 'react';

const METRICS = [
  { value: 1, prefix: '<', suffix: 's', label: 'Analysis latency' },
  { value: 200, prefix: '', suffix: '+', label: 'Telemetry signals processed per second' },
  { value: 0, prefix: '', suffix: '', label: 'Voice delivery', display: 'Real-time' },
  { value: 24, prefix: '', suffix: '/7', label: 'Your AI engineer never tires' },
];

const MetricsBar: React.FC = () => {
  const ref = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (!ref.current) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          obs.disconnect();
        }
      },
      { threshold: 0.3 }
    );
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);

  return (
    <section className="lp-metrics" ref={ref}>
      <div className="lp-metrics__grid">
        {METRICS.map((m, i) => (
          <div
            key={i}
            className={`lp-metrics__item ${inView ? 'lp-metrics__item--visible' : ''}`}
            style={{ '--item-delay': `${i * 0.15}s` } as React.CSSProperties}
          >
            <div className="lp-metrics__value">
              {m.display ? (
                <span>{m.display}</span>
              ) : (
                <>
                  {m.prefix}
                  <Counter target={m.value} active={inView} />
                  {m.suffix}
                </>
              )}
            </div>
            <div className="lp-metrics__label">{m.label}</div>
          </div>
        ))}
      </div>
    </section>
  );
};

/* ── Animated Counter ──────────────────────────── */
const Counter: React.FC<{ target: number; active: boolean }> = ({ target, active }) => {
  const [val, setVal] = useState(0);

  useEffect(() => {
    if (!active || target === 0) return;
    let current = 0;
    const step = Math.max(1, Math.ceil(target / 40));
    const interval = setInterval(() => {
      current += step;
      if (current >= target) {
        setVal(target);
        clearInterval(interval);
      } else {
        setVal(current);
      }
    }, 30);
    return () => clearInterval(interval);
  }, [active, target]);

  return <span>{val}</span>;
};

export default MetricsBar;
