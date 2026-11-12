import React from 'react';
import type { PhraseEvent } from './phrase-engine';

const clockFormatter = new Intl.DateTimeFormat(undefined, {
    hour: 'numeric', minute: 'numeric', second: 'numeric', fractionalSecondDigits: 3,
});

function PhraseResponse({ event }: { event: PhraseEvent }) {
    const speech = event.speech;
    const stages = [
        { label: 'Trigger', at: event.timestamp },
        { label: 'Play request', at: speech?.requestedAt },
        { label: 'Playback start', at: speech?.playbackStartedAt },
        { label: 'Audio end', at: speech?.finishedAt },
        { label: 'Phrase release', at: speech?.releaseAt },
    ];
    const outcome = speech?.outcome;
    const outcomes = { completed: 'Completed', replaced: 'Interrupted', rejected: 'Rejected', cancelled: 'Cancelled', failed: 'Failed' };

    return <div className="live-phrases__response">
        <span className="live-phrases__response-label">Phrase response</span>
        <ol aria-label="Phrase response timeline">
            {stages.map((stage, index) => <li key={stage.label} data-recorded={stage.at !== undefined}>
                {index > 0 && <span className="live-phrases__response-arrow" aria-hidden="true">→</span>}
                <div>
                    <strong>{stage.label}</strong>
                    {stage.at !== undefined ? <>
                        <time dateTime={new Date(stage.at).toISOString()}>{clockFormatter.format(stage.at)}</time>
                        <span>+{stage.at - event.timestamp} ms</span>
                    </> : <span>{!speech ? 'Unavailable' : speech.finishedAt !== undefined ? 'Not reached' : 'Pending'}</span>}
                    {stage.label === 'Audio end' && outcome && <span data-outcome={outcome}>{outcomes[outcome]}</span>}
                </div>
            </li>)}
        </ol>
    </div>;
}

export default React.memo(PhraseResponse);
