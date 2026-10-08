import React from 'react';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import { isOverlayRecord, isOverlayNonEmptyString } from 'views/floating-chat/overlay-renderer-validation';
import type { LivePhraseOverlaySnapshot } from '../LivePhraseOverlay';
import './LivePhraseDisplay.css';

// Keep phrase visuals here; the overlay shell only invokes this renderer.
export const LivePhraseDisplay = ({ snapshot }: { snapshot: LivePhraseOverlaySnapshot }) => (
    <section className="live-phrase-display" aria-label={`Live phrase: ${snapshot.category}`}>
        <header className="live-phrase-display__header">
            <span>Live phrase</span>
            <time dateTime={new Date(snapshot.timestamp).toISOString()}>
                {new Date(snapshot.timestamp).toLocaleTimeString()}
            </time>
        </header>
        <h3>{snapshot.category}</h3>
        <p>{snapshot.sentence}</p>
    </section>
);

export const livePhraseOverlayRenderer: AiOverlayRenderer<LivePhraseOverlaySnapshot> = {
    componentType: 'live_phrase',
    validateSnapshot: (snapshot): snapshot is LivePhraseOverlaySnapshot => (
        isOverlayRecord(snapshot)
        && typeof snapshot.eventId === 'number' && Number.isInteger(snapshot.eventId) && snapshot.eventId > 0
        && isOverlayNonEmptyString(snapshot.ruleId)
        && isOverlayNonEmptyString(snapshot.category)
        && isOverlayNonEmptyString(snapshot.sentence)
        && typeof snapshot.timestamp === 'number' && Number.isFinite(new Date(snapshot.timestamp).getTime())
    ),
    renderOverlay: (snapshot, status) => status === 'folded'
        ? snapshot.category
        : <LivePhraseDisplay snapshot={snapshot} />,
    dimensions: {
        expanded: { width: 420, height: 240 },
        folded: { width: 300, height: 58 },
        focus: { width: 420, height: 240 },
    },
};
