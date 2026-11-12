import React from 'react';
import { render, screen, within } from '@testing-library/react';
import PhraseScheduler from './PhraseScheduler';
import type { PhraseScheduleSnapshot } from './phrase-scheduler';

const schedule: PhraseScheduleSnapshot = { now: 1000, items: [
    { id: 'brake', kind: 'phrase', label: 'Brake.', priority: 3, category: 'brake', status: 'active', startAt: 800, endAt: 1800 },
    { id: 'turn', kind: 'phrase', label: 'Turn in', priority: 2, category: 'turn', status: 'next', startAt: 1800, endAt: 2800 },
    { id: 'throttle', kind: 'phrase', label: 'Squeeze gas.', priority: 1, category: 'throttle', status: 'queued', startAt: 2800, endAt: 3800, durationEstimated: true },
] };

it('shows the supplied execution order, priorities, active phrase and next phrase', () => {
    render(<PhraseScheduler schedule={schedule} enabled />);
    const cues = within(screen.getByRole('list', { name: 'Scheduled cues' })).getAllByRole('listitem');
    expect(cues.map((cue) => cue.getAttribute('aria-label'))).toEqual([
        expect.stringContaining('Brake. · Phrase · Priority 3 · Active'),
        expect.stringContaining('Turn in · Phrase · Priority 2 · Next'),
        expect.stringContaining('Squeeze gas. · Phrase · Priority 1 · Queued'),
    ]);
    expect(screen.getByText('Up next').parentElement).toHaveTextContent('Turn in');
    expect(screen.getByText(/Timing is estimated/)).toBeVisible();
    expect(cues[2]).toHaveAccessibleName(expect.stringContaining('Duration ~1.0 s'));
});

it('replaces the schedule when a prediction disappears', () => {
    const view = render(<PhraseScheduler schedule={schedule} enabled />);
    view.rerender(<PhraseScheduler schedule={{ now: 2000, items: [
        { ...schedule.items[1], status: 'active' },
    ] }} enabled />);
    expect(screen.queryByText('Brake.')).not.toBeInTheDocument();
    expect(screen.queryByText('Squeeze gas.')).not.toBeInTheDocument();
    expect(screen.queryByText('Up next')).not.toBeInTheDocument();
    expect(screen.queryByText('Next closure')).not.toBeInTheDocument();
    expect(screen.getAllByRole('listitem')).toHaveLength(1);
    expect(screen.getByText('1 scheduled')).toBeVisible();
});

it('distinguishes disabled detection from waiting without displaying sample cues', () => {
    const view = render(<PhraseScheduler schedule={{ now: 1000, items: [] }} enabled={false} />);
    expect(screen.getByText('Enable detection to see the live schedule.')).toBeVisible();
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
    view.rerender(<PhraseScheduler schedule={{ now: 1000, items: [] }} enabled />);
    expect(screen.getByText('Waiting for a predicted phrase.')).toBeVisible();
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
});
