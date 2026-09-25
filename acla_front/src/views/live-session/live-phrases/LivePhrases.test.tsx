import React from 'react';
import { act, render, screen, within } from '@testing-library/react';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import LivePhrases from './LivePhrases';
import { PHRASE_RULES } from './phrase-engine';
import { vision } from './test-fixtures';

describe('LivePhrases root connection', () => {
    beforeEach(() => { jest.useFakeTimers(); jest.setSystemTime(10000); });
    afterEach(() => { jest.useRealTimers(); });

    it('lists the entire catalog before the live root is available', () => {
        render(<LivePhrases name="live-phrases" />);
        const catalog = within(screen.getByRole('region', { name: 'Sentence catalog' }));
        PHRASE_RULES.forEach((rule) => expect(catalog.getByText(rule.sentence)).toBeVisible());
        expect(catalog.getAllByRole('listitem')).toHaveLength(PHRASE_RULES.length);
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
    });

    it('consumes only root events, expires stale input, resets history and releases subscriptions', () => {
        const telemetry = createLiveTelemetryStore();
        let detection: TrackVisionDetection | null = null;
        const listeners = new Set<() => void>();
        const stopTelemetry = jest.fn();
        const stopVision = jest.fn();
        const root = {
            getComponentName: () => 'live-session',
            subscribeTelemetry: jest.fn((listener) => {
                const unsubscribe = telemetry.subscribeEvents(listener);
                return () => { stopTelemetry(); unsubscribe(); };
            }),
            getTrackVisionDetection: () => detection,
            subscribeTrackVision: (listener: () => void) => {
                listeners.add(listener);
                return () => { stopVision(); listeners.delete(listener); };
            },
        };
        const Root = () => {
            const ref = React.useRef(root);
            useRegisterOperationComponentRef(ref);
            return null;
        };
        const view = render(<OperationComponentRefProvider><Root /><LivePhrases name="live-phrases" /></OperationComponentRefProvider>);
        const publish = (sequence: number) => telemetry.publishFrame({
            type: 'frame', game: 'acc', sequence, committedCount: 0, committedSequence: 0,
            sample: { Graphics_status: 2, Physics_speed_kmh: 100 },
        });
        act(() => {
            detection = vision(Date.now());
            listeners.forEach((listener) => listener());
            publish(1);
            jest.advanceTimersByTime(800);
            publish(2);
        });
        const output = within(screen.getByRole('region', { name: 'Triggered sentences' }));
        const sentence = 'Left-hand corner: you are on the inside; the opponent ahead is on the outside.';
        expect(output.getByText(sentence)).toBeInTheDocument();
        expect(output.queryByText('You are accelerating.')).not.toBeInTheDocument();
        const catalog = within(screen.getByRole('region', { name: 'Sentence catalog' }));
        PHRASE_RULES.forEach((rule) => expect(catalog.getByText(rule.sentence)).toBeVisible());
        act(() => { jest.advanceTimersByTime(2250); });
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
        expect(screen.getByText('Track Vision: Unavailable or stale')).toBeInTheDocument();
        act(() => { telemetry.resetSession(); });
        expect(output.queryByText(sentence)).not.toBeInTheDocument();
        view.unmount();
        expect(stopTelemetry).toHaveBeenCalledTimes(1);
        expect(stopVision).toHaveBeenCalledTimes(1);
        expect(listeners.size).toBe(0);
        expect(jest.getTimerCount()).toBe(0);
    });
});
