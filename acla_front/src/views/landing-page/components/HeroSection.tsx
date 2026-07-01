import React, { useMemo } from 'react';

const PARTICLE_COLORS = ['var(--lp-green)', 'var(--lp-red)', 'var(--lp-blue)'];

interface HeroSectionProps {
  onLoginClick: () => void;
}

const HeroSection: React.FC<HeroSectionProps> = ({ onLoginClick }) => {
  const particles = useMemo(
    () =>
      Array.from({ length: 60 }, (_, i) => ({
        id: i,
        top: Math.random() * 100,
        size: 1.5 + Math.random() * 3,
        color: PARTICLE_COLORS[Math.floor(Math.random() * 3)],
        duration: 6 + Math.random() * 12,
        delay: Math.random() * 8,
        opacity: 0.3 + Math.random() * 0.5,
      })),
    []
  );

  const scrollToSection2 = () => {
    const el = document.getElementById('timeline-section');
    el?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="lp-hero">
      {/* Particle field */}
      <div className="lp-hero__particles" aria-hidden="true">
        {particles.map((p) => (
          <div
            key={p.id}
            className="lp-hero__particle"
            style={
              {
                '--p-top': `${p.top}%`,
                '--p-size': `${p.size}px`,
                '--p-color': p.color,
                '--p-duration': `${p.duration}s`,
                '--p-delay': `${p.delay}s`,
                '--p-opacity': p.opacity,
              } as React.CSSProperties
            }
          />
        ))}
      </div>

      {/* Grid overlay */}
      <div className="lp-hero__grid" aria-hidden="true" />

      {/* Login button — top-right of hero */}
      <button className="lp-hero__login-btn" onClick={onLoginClick} type="button">
        Login
      </button>

      {/* Content */}
      <div className="lp-hero__content">
        <span className="lp-hero__logo-btn" aria-label="ACLA Home">
          ACLA
        </span>
        <h1 className="lp-hero__headline">
          Your AI Race Engineer.
          <br />
          <span className="lp-hero__headline--accent">In Real Time.</span>
        </h1>
        <p className="lp-hero__subline">
          Sub-second analysis. Voice-guided. Hands-free.
        </p>
        <button className="lp-hero__cta" onClick={scrollToSection2} type="button">
          See It In Action
          <span className="lp-hero__cta-arrow">↓</span>
        </button>
      </div>

      {/* Helmet icon (bottom-right) */}
      <div className="lp-hero__helmet" aria-hidden="true">
        <svg viewBox="0 0 64 64" fill="none" width="80" height="80">
          <path
            d="M32 4C18 4 8 16 8 28v8c0 6 4 12 10 14l4 2h20l4-2c6-2 10-8 10-14v-8C56 16 46 4 32 4z"
            stroke="var(--lp-green)"
            strokeWidth="1.5"
            strokeLinecap="round"
            opacity="0.5"
          />
          <path
            d="M12 30h40"
            stroke="var(--lp-green)"
            strokeWidth="1"
            opacity="0.3"
          />
          <rect x="10" y="30" width="18" height="6" rx="2"
            stroke="var(--lp-green)"
            strokeWidth="1"
            fill="none"
            opacity="0.4"
          />
        </svg>
      </div>
    </section>
  );
};

export default HeroSection;
