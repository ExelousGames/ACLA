import React from 'react';
import { act, render, screen, within } from '@testing-library/react';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import LivePhrases from './LivePhrases';
import { PHRASE_RULES } from './phrase-engine';
import { circuitMap, vision } from './test-fixtures';

describe('LivePhrases root connection', () => {
    beforeEach(() => { jest.useFakeTimers(); jest.setSystemTime(10000); });
    afterEach(() => { jest.useRealTimers(); });

    it('lists the entire catalog before the live root is available', () => {
        render(<LivePhrases name="live-phrases" />);
        const catalog = within(screen.getByRole('region', { name: 'Sentence catalog' }));
        PHRASE_RULES.forEach((rule) => {
            expect(catalog.getByText(rule.sentence)).toBeVisible();
            const conditions = within(catalog.getByRole('list', { name: `${rule.category} conditions` }));
            expect(conditions.getAllByRole('listitem')).toHaveLength(rule.conditions.length);
            expect(conditions.getAllByText('Missing input')).toHaveLength(rule.conditions.length);
            conditions.getAllByRole('listitem').forEach((condition) => expect(condition).toHaveAttribute('data-condition-fit', 'false'));
        });
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
        expect(screen.getByText('Live Map: Waiting for a tagged circuit map')).toBeInTheDocument();
    });

    it('consumes only root events, expires stale input, resets history and releases subscriptions', () => {
        const telemetry = createLiveTelemetryStore();
        let detection: TrackVisionDetection | null = null;
        const listeners = new Set<() => void>();
        const stopTelemetry = jest.fn();
        const stopVision = jest.fn();
        const stopMap = jest.fn();
        let map: ReturnType<typeof circuitMap> | null = null;
        let notifyMap!: () => void;
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
            getLiveCircuitMap: () => map,
            subscribeLiveCircuitMap: (listener: () => void) => { notifyMap = listener; return stopMap; },
        };
        const Root = () => {
            const ref = React.useRef(root);
            useRegisterOperationComponentRef(ref);
            return null;
        };
        const view = render(<OperationComponentRefProvider><Root /><LivePhrases name="live-phrases" /></OperationComponentRefProvider>);
        const publish = (sequence: number, speed = 100) => telemetry.publishFrame({
            type: 'frame', game: 'acc', sequence, committedCount: 0, committedSequence: 0,
            sample: { Graphics_status: 2, Physics_speed_kmh: speed, Graphics_normalized_car_position: 0.11 },
        });
        act(() => {
            detection = vision(Date.now());
            listeners.forEach((listener) => listener());
            map = circuitMap();
            notifyMap();
            publish(1);
            jest.advanceTimersByTime(800);
            publish(2);
        });
        const output = within(screen.getByRole('region', { name: 'Triggered sentences' }));
        const sentence = PHRASE_RULES.find((rule) => rule.id === 'inside-outbraking')!.sentence;
        expect(output.getByText(sentence)).toBeInTheDocument();
        expect(output.queryByText('You are accelerating.')).not.toBeInTheDocument();
        expect(screen.getByText('Live Map: Ready')).toBeInTheDocument();
        expect(screen.getByText('Current section: slow corner · entry')).toBeInTheDocument();
        expect(screen.getByText('Corner shape: bend')).toBeInTheDocument();
        const conditions = within(screen.getByRole('list', { name: 'Outbraking on the inside conditions' }));
        const conditionCount = PHRASE_RULES.find((rule) => rule.id === 'inside-outbraking')!.conditions.length;
        expect(conditions.getAllByText('Met')).toHaveLength(conditionCount);
        conditions.getAllByRole('listitem').forEach((condition) => expect(condition).toHaveAttribute('data-condition-fit', 'true'));
        act(() => { publish(3, 0); });
        const speedCondition = within(conditions.getByText('Speed (km/h) >= 30').closest('li')!);
        expect(speedCondition.getByText('Not met')).toBeVisible();
        expect(conditions.getAllByText('Met')).toHaveLength(conditionCount - 1);
        act(() => { publish(4); });
        expect(speedCondition.getByText('Met')).toBeVisible();
        act(() => {
            map = circuitMap('slow', true);
            map.centerline_tags!.push({ id: 'sequence', label: 'consecutive corners', start_position: 0.1, end_position: 0.31 });
            notifyMap();
        });
        expect(screen.getByText('Corner sequence: alternating · 1 of 2 corners')).toBeInTheDocument();
        const catalog = within(screen.getByRole('region', { name: 'Sentence catalog' }));
        PHRASE_RULES.forEach((rule) => expect(catalog.getByText(rule.sentence)).toBeVisible());
        act(() => { jest.advanceTimersByTime(2250); });
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
        expect(screen.getByText('Track Vision: Unavailable or stale')).toBeInTheDocument();
        expect(conditions.getAllByText('Missing input')).toHaveLength(conditionCount);
        conditions.getAllByRole('listitem').forEach((condition) => expect(condition).toHaveAttribute('data-condition-fit', 'false'));
        act(() => { telemetry.resetSession(); });
        expect(output.queryByText(sentence)).not.toBeInTheDocument();
        act(() => { map = null; notifyMap(); });
        expect(screen.getByText('Live Map: Waiting for a tagged circuit map')).toBeInTheDocument();
        view.unmount();
        expect(stopTelemetry).toHaveBeenCalledTimes(1);
        expect(stopVision).toHaveBeenCalledTimes(1);
        expect(stopMap).toHaveBeenCalledTimes(1);
        expect(listeners.size).toBe(0);
        expect(jest.getTimerCount()).toBe(0);
    });
});
