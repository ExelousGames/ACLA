import React from 'react';
import type { PhraseEvent } from './phrase-engine';
import PhraseResponse from './PhraseResponse';

interface PhraseHistoryProps {
    events: readonly PhraseEvent[];
    emptyMessage: string;
}

function PhraseHistory({ events, emptyMessage }: PhraseHistoryProps) {
    return <section aria-label="Phrase history" className="live-phrases__output">
        <h3>Phrase history</h3>
        {events.length > 0 && <p className="live-phrases__hint">Response times are measured from the trigger.</p>}
        {events.length === 0
            ? <p>{emptyMessage}</p>
            : <ol>{events.slice().reverse().map((event) => (
                <li key={event.id}>
                    <span>{event.sentence}</span>
                    <PhraseResponse event={event} />
                </li>
            ))}</ol>}
    </section>;
}

// The engine replaces changed events; telemetry can also publish a new array of unchanged events.
export default React.memo(PhraseHistory, (previous, next) => previous.emptyMessage === next.emptyMessage
    && (previous.events === next.events || (previous.events.length === next.events.length
        && previous.events.every((event, index) => event === next.events[index]))));
