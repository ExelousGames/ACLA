import React from 'react';
import { render, screen, within } from '@testing-library/react';
import type { PhraseEvent } from './phrase-engine';
import PhraseHistory from './PhraseHistory';

afterEach(() => jest.restoreAllMocks());

it('skips unchanged history and only formats the row whose phrase state changes', () => {
    const completed: PhraseEvent = {
        id: 1, ruleId: 'brake', sentence: 'Brake.', timestamp: 1000,
        speech: { reactionAllowanceMs: 0, requestedAt: 1005, playbackStartedAt: 1080, finishedAt: 1900, releaseAt: 1900, outcome: 'completed' },
    };
    const triggered: PhraseEvent = {
        id: 2, ruleId: 'throttle', sentence: 'Squeeze gas.', timestamp: 2000, speech: { reactionAllowanceMs: 0 },
    };
    const emptyMessage = 'No phrases yet.';
    const view = render(<PhraseHistory events={[completed]} emptyMessage={emptyMessage} />);
    const formatTimestamp = jest.spyOn(Date.prototype, 'toISOString');

    // An evaluation may return a different array containing the same immutable events.
    view.rerender(<PhraseHistory events={[completed]} emptyMessage={emptyMessage} />);
    expect(formatTimestamp).not.toHaveBeenCalled();

    view.rerender(<PhraseHistory events={[completed, triggered]} emptyMessage={emptyMessage} />);
    expect(formatTimestamp).toHaveBeenCalledTimes(1);
    const timelines = screen.getAllByRole('list', { name: 'Phrase response timeline' });
    expect(within(timelines[0]).getAllByText('Pending')).toHaveLength(4);
    expect(within(timelines[1]).getByText('Completed')).toBeInTheDocument();

    formatTimestamp.mockClear();
    const requested = { ...triggered, speech: { ...triggered.speech!, requestedAt: 2005 } };
    view.rerender(<PhraseHistory events={[completed, requested]} emptyMessage={emptyMessage} />);
    expect(formatTimestamp).toHaveBeenCalledTimes(2);
    expect(within(timelines[0]).getByText('+5 ms')).toBeInTheDocument();

    view.rerender(<PhraseHistory events={[]} emptyMessage={emptyMessage} />);
    expect(screen.queryByRole('list', { name: 'Phrase response timeline' })).not.toBeInTheDocument();
    expect(screen.getByText(emptyMessage)).toBeInTheDocument();
});
