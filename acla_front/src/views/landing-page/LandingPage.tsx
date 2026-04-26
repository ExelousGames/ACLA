import React, { useState, useEffect } from 'react';
import './landing-page.css';
import Preloader from './components/Preloader';
import HeroSection from './components/HeroSection';
import TimelineSection from './components/TimelineSection';
import LapSection from './components/LapSection';
import MetricsBar from './components/MetricsBar';
import CTASection from './components/CTASection';
import LoginModal from './components/LoginModal';

const LandingPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [showNav, setShowNav] = useState(false);
  const [loginModalOpen, setLoginModalOpen] = useState(false);

  /* Preloader timer */
  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 2800);
    return () => clearTimeout(timer);
  }, []);

  /* Show nav after scrolling past hero */
  useEffect(() => {
    const handleScroll = () => {
      setShowNav(window.scrollY > window.innerHeight * 0.6);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  if (loading) {
    return <Preloader />;
  }

  return (
    <div className="landing-page">
      {/* Fixed nav — appears on scroll */}
      <nav className={`lp-nav ${showNav ? 'lp-nav--visible' : ''}`}>
        <span className="lp-nav__logo">ACLA</span>
        <button className="lp-nav__login" onClick={() => setLoginModalOpen(true)} type="button">
          Login
        </button>
      </nav>

      <HeroSection onLoginClick={() => setLoginModalOpen(true)} />
      <TimelineSection />
      <LapSection />
      <MetricsBar />
      <CTASection />

      {loginModalOpen && <LoginModal onClose={() => setLoginModalOpen(false)} />}
    </div>
  );
};

export default LandingPage;
