import React, { useEffect, useMemo, useRef, useState } from 'react';

interface Msg {
  who: 'driver' | 'acla';
  name: string;
  stamp: string;
  text: string;
}

const CONVO_SCRIPT: Msg[] = [
  { who: 'driver', name: 'YOU',  stamp: '00:14:02', text: 'How are my front tires holding up?' },
  { who: 'acla',   name: 'ACLA', stamp: '00:14:03', text: 'Front-left at 96°C, light graining. Soften your turn-in through T7 and they’ll recover in two laps.' },
  { who: 'driver', name: 'YOU',  stamp: '00:14:21', text: 'Can I push for the leader before the pit window?' },
  { who: 'acla',   name: 'ACLA', stamp: '00:14:22', text: 'Gap is 1.4s and closing. You have the pace — save fuel through S1, then attack into T6.' },
  { who: 'driver', name: 'YOU',  stamp: '00:14:48', text: 'What’s the weather looking like in S3?' },
  { who: 'acla',   name: 'ACLA', stamp: '00:14:49', text: 'Light drizzle starting next lap. Box for inters is on the table — I’ll call it if it intensifies.' },
];

const ConversationSection: React.FC = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(0);
  const [typing, setTyping] = useState(false);
  const [active, setActive] = useState(false);
  const [clock, setClock] = useState('00:14:02');

  useEffect(() => {
    if (!sectionRef.current) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setActive(true); },
      { threshold: 0.25 }
    );
    obs.observe(sectionRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const run = async () => {
      while (!cancelled) {
        for (let i = 0; i < CONVO_SCRIPT.length; i++) {
          if (cancelled) return;
          if (CONVO_SCRIPT[i].who === 'acla') {
            setTyping(true);
            await new Promise(r => setTimeout(r, 900));
            if (cancelled) return;
            setTyping(false);
          }
          setShown(i + 1);
          setClock(CONVO_SCRIPT[i].stamp);
          await new Promise(r => setTimeout(r, CONVO_SCRIPT[i].who === 'driver' ? 1400 : 2600));
        }
        await new Promise(r => setTimeout(r, 3200));
        if (cancelled) return;
        setShown(0);
        setTyping(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [active]);

  const messages = CONVO_SCRIPT.slice(0, shown);
  const lastDriverSpoke = messages.length > 0 && messages[messages.length - 1].who === 'driver';

  const waveBars = useMemo(
    () => Array.from({ length: 24 }, (_, i) => ({
      delay: `${(i % 6) * 0.08}s`,
      duration: `${0.7 + (i % 5) * 0.1}s`,
    })),
    []
  );

  return (
    <section className="lp-convo" ref={sectionRef}>
      <div className="lp-convo__grid-bg" aria-hidden="true" />
      <div className="lp-convo__inner">
        <div className="lp-convo__header">
          <span className="lp-convo__eyebrow">
            <span className="lp-convo__eyebrow-dot" />
            Even better feature
          </span>
          <h2 className="lp-convo__heading">
            Conversational <span className="lp-convo__heading-accent">AI</span>
          </h2>
          <p className="lp-convo__sub">
            Talk to ACLA while you race. Ask questions on the fly — tires, fuel, strategy, the car ahead.
            Hands on the wheel, eyes on the apex.
          </p>
        </div>

        <div className="lp-convo__stage">
          <div className="lp-convo__mic-panel">
            <div className="lp-convo__mic-header">
              <span className="lp-convo__mic-channel">
                <span className="lp-convo__eyebrow-dot" />
                CH-1 · OPEN
              </span>
              <span>VOICE LINK</span>
            </div>

            <div className="lp-convo__mic-visual">
              <span className="lp-convo__mic-ring" />
              <span className="lp-convo__mic-ring" />
              <span className="lp-convo__mic-ring" />
              <div className="lp-convo__mic-core">
                <svg viewBox="0 0 48 48" width="44" height="44" fill="none">
                  <rect
                    x="18" y="6" width="12" height="22" rx="6"
                    stroke="var(--lp-green)" strokeWidth="2" fill="rgba(0,230,118,0.08)"
                  />
                  <path
                    d="M10 22c0 7.7 6.3 14 14 14s14-6.3 14-14"
                    stroke="var(--lp-green)" strokeWidth="2" strokeLinecap="round"
                  />
                  <line x1="24" y1="36" x2="24" y2="42" stroke="var(--lp-green)" strokeWidth="2" />
                  <line x1="17" y1="42" x2="31" y2="42" stroke="var(--lp-green)" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </div>
            </div>

            <div className="lp-convo__mic-status">
              {lastDriverSpoke ? 'ACLA' : 'Driver'}
              <b>{typing ? 'THINKING…' : (lastDriverSpoke ? 'RESPONDING' : 'LISTENING')}</b>
            </div>

            <div className="lp-convo__mic-wave" aria-hidden="true">
              {waveBars.map((b, i) => (
                <span
                  key={i}
                  className="lp-convo__mic-wave-bar"
                  style={{ animationDelay: b.delay, animationDuration: b.duration }}
                />
              ))}
            </div>

            <div className="lp-convo__hint">
              Push <kbd>PTT</kbd> or just say <kbd>&ldquo;Hey ACLA&rdquo;</kbd><br />
              No menus. No screens. Just talk.
            </div>
          </div>

          <div className="lp-convo__transcript">
            <div className="lp-convo__transcript-head">
              <span className="lp-convo__transcript-title">
                <span className="lp-convo__eyebrow-dot" />
                LIVE TRANSCRIPT · LAP 14
              </span>
              <span className="lp-convo__transcript-time">{clock}</span>
            </div>

            <div className="lp-convo__msgs">
              {messages.map((m, i) => (
                <div key={`${shown}-${i}`} className={`lp-convo__msg lp-convo__msg--${m.who}`}>
                  <div className="lp-convo__msg-avatar">{m.who === 'driver' ? 'YOU' : 'AI'}</div>
                  <div className="lp-convo__msg-body">
                    <div className="lp-convo__msg-meta">
                      <span className="lp-convo__msg-who">{m.name}</span>
                      <span className="lp-convo__msg-stamp">{m.stamp}</span>
                    </div>
                    <div className="lp-convo__msg-text">{m.text}</div>
                  </div>
                </div>
              ))}
              {typing && (
                <div className="lp-convo__msg lp-convo__msg--acla">
                  <div className="lp-convo__msg-avatar">AI</div>
                  <div className="lp-convo__msg-body">
                    <div className="lp-convo__msg-meta">
                      <span className="lp-convo__msg-who">ACLA</span>
                    </div>
                    <div className="lp-convo__typing">
                      <span className="lp-convo__typing-dot" />
                      <span className="lp-convo__typing-dot" />
                      <span className="lp-convo__typing-dot" />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="lp-convo__pills">
          <span className="lp-convo__pill">How&apos;s my fuel?</span>
          <span className="lp-convo__pill">Best line through T3?</span>
          <span className="lp-convo__pill">Pit or stay out?</span>
          <span className="lp-convo__pill">Who&apos;s behind me?</span>
          <span className="lp-convo__pill">Bring it home safe</span>
        </div>
      </div>
    </section>
  );
};

export default ConversationSection;
