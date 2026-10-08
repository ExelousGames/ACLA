import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import LivePhrases from './LivePhrases';
import { PHRASE_RULES, PhraseCondition, PhraseEngine, conditionGroup } from './phrase-engine';
import { circuitMap, vision } from './test-fixtures';

jest.mock('components/tts', () => ({ synthesizeTts: jest.fn(() => new Promise(() => undefined)) }));

describe('LivePhrases root connection', () => {
    beforeEach(() => { jest.useFakeTimers(); jest.setSystemTime(10000); });
    afterEach(() => { jest.useRealTimers(); jest.restoreAllMocks(); });

    it('lists the entire catalog before the live root is available', () => {
        render(<LivePhrases name="live-phrases" />);
        expect(screen.getByRole('button', { name: 'Enable detection' })).toBeVisible();
        expect(screen.getByText('Detection: Disabled')).toBeInTheDocument();
        expect(jest.getTimerCount()).toBe(0);
        const catalog = within(screen.getByRole('region', { name: 'Sentence catalog' }));
        PHRASE_RULES.forEach((rule) => {
            expect(catalog.getByText(rule.sentence)).toBeVisible();
            const conditions = within(catalog.getByRole('list', { name: `${rule.category} conditions` }));
            expect(conditions.getAllByRole('listitem')).toHaveLength(rule.conditions.length);
            expect(conditions.getAllByText('Missing input')).toHaveLength(rule.conditions.length);
            conditions.getAllByRole('listitem').forEach((condition) => expect(condition).toHaveAttribute('data-condition-fit', 'false'));
            expect(conditions.getAllByText('AND')).toHaveLength(rule.conditions.length - 1);
            expect(within(conditions.getAllByRole('listitem')[0]).queryByText('AND')).not.toBeInTheDocument();
        });
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
        expect(screen.getByText('Live Map: Waiting for a tagged circuit map')).toBeInTheDocument();
        const positions = within(catalog.getByRole('list', { name: 'Outbraking on the inside conditions' }));
        for (const subject of ['Player', 'Opponent']) {
            expect(positions.getByText(`${subject} in a left or right turn corner`)).toBeVisible();
            expect(positions.getByText(`${subject} near the left edge or middle of the track or right edge`)).toBeVisible();
        }
    });

    it('displays mixed connectors from the evaluated conditions in order', () => {
        const rules = PHRASE_RULES.map((rule, index) => index === 0 ? {
            ...rule,
            conditions: [
                new PhraseCondition('speed', '>=', 30),
                new PhraseCondition('carAhead', '=', 1, undefined, 'or'),
                new PhraseCondition('phase', '=', 'entry'),
            ],
        } : rule);
        const snapshot = new PhraseEngine(rules).evaluate(Date.now());
        jest.spyOn(PhraseEngine.prototype, 'evaluate').mockReturnValue(snapshot);
        render(<LivePhrases name="live-phrases" />);
        const conditions = within(screen.getByRole('list', { name: `${rules[0].category} conditions` }));
        const items = conditions.getAllByRole('listitem');
        expect(items).toHaveLength(3);
        expect(within(items[0]).queryByText(/^(AND|OR)$/)).not.toBeInTheDocument();
        expect(within(items[1]).getByText('OR')).toBeVisible();
        expect(within(items[2]).getByText('AND')).toBeVisible();
        expect(conditions.getAllByText('Missing input')).toHaveLength(3);
        expect(screen.getByText(/AND is evaluated before OR/)).toBeVisible();
    });

    it('displays nested group boundaries, connectors and each independent status', () => {
        // A OR (B AND (C OR missing)): both groups match while A and the missing leaf do not.
        const rules = PHRASE_RULES.map((rule, index) => index === 0 ? {
            ...rule,
            conditions: [
                new PhraseCondition('speed', '>=', 120),
                conditionGroup([
                    new PhraseCondition('speed', '>=', 30, undefined, 'or'),
                    conditionGroup([
                        new PhraseCondition('speed', '<=', 110, undefined, 'or'),
                        new PhraseCondition('carAhead', '=', 1, undefined, 'or'),
                    ]),
                ], 'or'),
            ],
        } : rule);
        const sample = { Physics_speed_kmh: 100 };
        const snapshot = new PhraseEngine(rules).receiveTelemetry({
            type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
            committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
            update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 },
        }, Date.now());
        jest.spyOn(PhraseEngine.prototype, 'evaluate').mockReturnValue(snapshot);
        render(<LivePhrases name="live-phrases" />);
        const list = screen.getByRole('list', { name: `${rules[0].category} conditions` });
        const conditions = within(list);
        const items = conditions.getAllByRole('listitem');
        const groups = conditions.getAllByRole('group', { name: 'Condition group' });
        expect(groups).toHaveLength(2);
        expect(groups[0]).toContainElement(groups[1]);
        expect(items).toHaveLength(6);
        expect(within(items[1]).getAllByText('OR')).toHaveLength(2);
        expect(within(groups[0]).getAllByText('OR')).toHaveLength(1);
        expect(within(items[3]).getByText('AND')).toBeVisible();
        expect(within(groups[1]).queryByText('AND')).not.toBeInTheDocument();
        expect(within(groups[0]).getAllByText('Met')).toHaveLength(4);
        expect(within(groups[1]).getAllByText('Met')).toHaveLength(2);
        for (const item of [items[1], items[3]]) {
            expect(item).toHaveAttribute('data-condition-fit', 'true');
            expect(item).toHaveAttribute('data-input-missing', 'false');
        }
        for (const item of [items[2], items[4]]) {
            expect(within(item).queryByText(/^(AND|OR)$/)).not.toBeInTheDocument();
        }
        expect(conditions.getAllByText('(')).toHaveLength(2);
        expect(conditions.getAllByText(')')).toHaveLength(2);
        const failed = within(items[0]);
        expect(failed.getByText('Speed (km/h) >= 120')).toBeVisible();
        expect(failed.getByText('Not met')).toBeVisible();
        const missing = within(items[5]);
        expect(missing.getByText('Opponent ahead on visible track = 1')).toBeVisible();
        expect(missing.getByText('Missing input')).toBeVisible();
        expect(missing.getByText('OR')).toBeVisible();
        const firstGuide = within(screen.getByRole('region', { name: 'Sentence catalog' })).getAllByRole('listitem')[0];
        expect(within(firstGuide).queryByText(/Waiting for:/)).not.toBeInTheDocument();
        expect(screen.getByText(/Parenthesized groups are evaluated first/)).toBeVisible();
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
        expect(root.subscribeTelemetry).not.toHaveBeenCalled();
        expect(listeners.size).toBe(0);
        act(() => { publish(0); });
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Enable detection' }));
        expect(screen.getByText('Detection: Enabled')).toBeInTheDocument();
        act(() => {
            detection = { ...vision(Date.now()), geometry: null };
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
        fireEvent.click(screen.getByRole('button', { name: 'Disable detection' }));
        expect(screen.getByText('Detection: Disabled')).toBeInTheDocument();
        expect(output.queryByText(sentence)).not.toBeInTheDocument();
        expect(stopTelemetry).toHaveBeenCalledTimes(1);
        expect(stopVision).toHaveBeenCalledTimes(1);
        expect(stopMap).toHaveBeenCalledTimes(1);
        expect(listeners.size).toBe(0);
        expect(jest.getTimerCount()).toBe(0);
        act(() => { publish(5); jest.advanceTimersByTime(800); });
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
        expect(output.queryByText(sentence)).not.toBeInTheDocument();
        detection = { ...vision(Date.now()), geometry: null };
        fireEvent.click(screen.getByRole('button', { name: 'Enable detection' }));
        expect(screen.getByText('Telemetry: Waiting for live data')).toBeInTheDocument();
        act(() => { publish(6); });
        expect(output.queryByText(sentence)).not.toBeInTheDocument();
        act(() => { jest.advanceTimersByTime(800); publish(7); });
        expect(output.getByText(sentence)).toBeInTheDocument();
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
        expect(stopTelemetry).toHaveBeenCalledTimes(2);
        expect(stopVision).toHaveBeenCalledTimes(2);
        expect(stopMap).toHaveBeenCalledTimes(2);
        expect(listeners.size).toBe(0);
        expect(jest.getTimerCount()).toBe(0);
    });
});
