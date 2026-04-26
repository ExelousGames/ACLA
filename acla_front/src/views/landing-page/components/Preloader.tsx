import React from 'react';

const Preloader: React.FC = () => {
  return (
    <div className="lp-preloader">
      <div className="lp-preloader__track">
        <svg viewBox="0 0 800 400" className="lp-preloader__svg">
          <path
            className="lp-preloader__path"
            d="M 300,60 L 550,60 Q 700,60 700,180 L 700,220 Q 700,340 550,340 L 250,340 Q 100,340 100,220 L 100,180 Q 100,60 250,60 Z"
            fill="none"
            stroke="var(--lp-green)"
            strokeWidth="2"
            strokeLinecap="round"
          />
          {/* Start/finish line */}
          <line
            className="lp-preloader__finish"
            x1="300" y1="50" x2="300" y2="70"
            stroke="var(--lp-white)"
            strokeWidth="3"
          />
        </svg>
      </div>
      <div className="lp-preloader__logo">ACLA</div>
      <div className="lp-preloader__tagline">AI Race Engineer</div>
    </div>
  );
};

export default Preloader;
