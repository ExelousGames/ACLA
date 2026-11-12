import React from 'react';
import type { PhraseScheduleSnapshot } from './phrase-scheduler';
import './PhraseScheduler.css';

const seconds = (milliseconds: number) => `${(milliseconds / 1000).toFixed(1)} s`;
const statuses = { active: 'Active', next: 'Next', queued: 'Queued' };

const PhraseScheduler = ({ schedule, enabled }: { schedule: PhraseScheduleSnapshot; enabled: boolean }) => {
    const { now, items } = schedule;
    const future = Math.max(4000, ...items.map((item) => item.endAt - now + 500));
    const past = future / 4;
    const start = now - past;
    const span = past + future;
    const position = (timestamp: number) => Math.max(0, Math.min(100, (timestamp - start) / span * 100));
    const active = items.find((item) => item.status === 'active');
    const next = items.find((item) => item.status === 'next');

    return (
        <section className="phrase-scheduler" aria-label="Phrase scheduler">
            <header className="phrase-scheduler__header">
                <div>
                    <h3>Scheduler</h3>
                    <p>Higher priority goes first when cues are ready together.</p>
                </div>
                <span className="phrase-scheduler__count">{items.length} scheduled</span>
            </header>
            <div className="phrase-scheduler__legend" aria-label="Default priorities">
                <span data-category="brake">Brake <b>3</b></span>
                <span data-category="turn">Turn <b>2</b></span>
                <span data-category="throttle">Throttle <b>1</b></span>
            </div>
            {items.length > 0 ? <>
                <div className="phrase-scheduler__summary">
                    {active && <span><b>Active</b> {active.label}</span>}
                    {next && <span><b>Up next</b> {next.label}</span>}
                </div>
                <div className="phrase-scheduler__scroll" tabIndex={0} aria-label="Scheduled cue timeline">
                    <div className="phrase-scheduler__timeline">
                        <div className="phrase-scheduler__axis" aria-hidden="true">
                            {[0, 1, 2, 3, 4].map((tick) => <span key={tick} style={{ left: `${tick * 20 + 20}%` }}>
                                {tick === 0 ? 'Now' : `+${seconds(future * tick / 4)}`}
                            </span>)}
                        </div>
                        <div className="phrase-scheduler__now" aria-hidden="true"><span>★</span></div>
                        <ol className="phrase-scheduler__lanes" aria-label="Scheduled cues">
                            {items.map((item) => {
                                const left = position(item.startAt);
                                const width = Math.max(0, position(item.endAt) - left);
                                const duration = `${item.durationEstimated ? '~' : ''}${seconds(Math.max(0, item.endAt - item.startAt))}`;
                                const detail = `Phrase · Priority ${item.priority} · ${statuses[item.status]}`;
                                const timing = item.startAt <= now ? 'Starts now or already active' : `Starts in ${seconds(item.startAt - now)}`;
                                return <li key={item.id} className="phrase-scheduler__lane"
                                    aria-label={`${item.label} · ${detail} · ${timing} · Duration ${duration}`}>
                                    <div className="phrase-scheduler__row-label">
                                        <span>{item.label}</span>
                                        <span>P{item.priority} · {statuses[item.status]}</span>
                                    </div>
                                    <div className="phrase-scheduler__track">
                                        <div className="phrase-scheduler__bar" data-category={item.category} data-status={item.status}
                                            style={{ left: `${left}%`, width: `${width}%` }}
                                            title={`${item.label} · ${detail} · ${timing} · ${duration}${item.predictedAt !== undefined
                                                ? ` · Predicted in ${seconds(Math.max(0, item.predictedAt - now))}` : ''}`}>
                                            <span>{duration}</span><i aria-hidden="true" />
                                        </div>
                                    </div>
                                </li>;
                            })}
                        </ol>
                    </div>
                </div>
                <p className="phrase-scheduler__hint">Timing is estimated from prepared speech. A phrase finishes when its audio ends or its exit condition matches. Queued phrases are checked again before they trigger.</p>
            </> : <div className="phrase-scheduler__empty">
                <span aria-hidden="true">◇</span>
                <p>{enabled ? 'Waiting for a predicted phrase.' : 'Enable detection to see the live schedule.'}</p>
            </div>}
        </section>
    );
};

export default PhraseScheduler;
