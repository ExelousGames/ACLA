import React from 'react';
import { useNavigate } from 'react-router-dom';

const CTASection: React.FC = () => {
  const navigate = useNavigate();

  return (
    <section className="lp-cta">
      <div className="lp-cta__glow" aria-hidden="true" />
      <div className="lp-cta__content">
        <h2 className="lp-cta__headline">
          Stop Guessing.<br />
          <span className="lp-cta__headline--accent">Start Racing.</span>
        </h2>
        <p className="lp-cta__subtext">
          Let your AI race engineer handle the thinking.
        </p>
        <div className="lp-cta__buttons">
          <button
            className="lp-cta__btn lp-cta__btn--primary"
            onClick={() => navigate('/register')}
            type="button"
          >
            Get Early Access
          </button>
          <button
            className="lp-cta__btn lp-cta__btn--secondary"
            type="button"
            onClick={() => {
              const hero = document.getElementById('timeline-section');
              hero?.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            Watch Demo
          </button>
        </div>
      </div>
    </section>
  );
};

export default CTASection;
