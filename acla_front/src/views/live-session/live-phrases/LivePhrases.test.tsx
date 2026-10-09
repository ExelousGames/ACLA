import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { OperationComponentRefProvider, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import LivePhrases from './LivePhrases';
import { PHRASE_DEFINITIONS, PhraseCondition, PhraseEngine, conditionGroup } from './phrase-engine';
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
        PHRASE_DEFINITIONS.forEach((rule) => {
            const summary = catalog.getByLabelText(`${rule.name} closure`);
            expect(summary).toBeVisible();
            expect(within(summary).getByText('Missing input')).toBeVisible();
            expect(catalog.getByText(rule.sentence)).not.toBeVisible();
            fireEvent.click(summary);
            fireEvent.click(catalog.getByLabelText(`${rule.name} entry conditions`));
            const conditions = within(catalog.getByRole('list', { name: `${rule.name} conditions` }));
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

    it('renders closure and action metadata from the engine snapshot', () => {
        const engine = new PhraseEngine([{ ...PHRASE_DEFINITIONS[0], additionalActions: undefined, name: 'Custom guide', description: 'Custom guide purpose.' }]);
        const snapshot = engine.evaluate(Date.now());
        snapshot.root = { ...snapshot.root, name: 'Custom root', description: 'Custom selection purpose.' };
        snapshot.root.children[0].children = [
            { ...snapshot.root.children[0].children[0], name: 'Custom action', description: 'Custom action purpose.' },
            { ...snapshot.root.children[0].children[1], name: 'Custom exit', description: 'Custom exit purpose.' },
        ];
        jest.spyOn(PhraseEngine.prototype, 'evaluate').mockReturnValue(snapshot);
        render(<LivePhrases name="live-phrases" />);
        const catalog = within(screen.getByRole('region', { name: 'Sentence catalog' }));
        expect(catalog.getByRole('heading', { name: 'Custom root closure (1 children)' })).toBeVisible();
        expect(catalog.getByText('Custom selection purpose.')).toBeVisible();
        expect(catalog.getByText('Custom guide')).toBeVisible();
        expect(catalog.getByText('Custom guide purpose.')).not.toBeVisible();
        fireEvent.click(catalog.getByLabelText('Custom guide closure'));
        expect(catalog.getByText('Custom guide purpose.')).toBeVisible();
        const actions = within(catalog.getByRole('list', { name: 'Custom guide children' }));
        expect(actions.getByText('Custom action')).toBeVisible();
        expect(actions.getByText('Custom action purpose.')).not.toBeVisible();
        fireEvent.click(actions.getByLabelText('Custom guide action 1: Custom action'));
        expect(actions.getByText('Custom action purpose.')).toBeVisible();
        expect(actions.getByText('Custom exit')).toBeVisible();
        expect(actions.getByText('Custom exit purpose.')).not.toBeVisible();
        fireEvent.click(actions.getByLabelText('Custom guide action 2: Custom exit'));
        expect(actions.getByText('Custom exit purpose.')).toBeVisible();
        fireEvent.click(actions.getByLabelText('Custom guide action 1: Custom action'));
        expect(actions.getByText('Custom action purpose.')).not.toBeVisible();
        expect(actions.getByText('Custom exit purpose.')).toBeVisible();
        fireEvent.click(catalog.getByLabelText('Custom root closure'));
        expect(catalog.getByText('Custom guide')).not.toBeVisible();
        expect(screen.getByLabelText('Closure state')).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Triggered sentences' })).toBeVisible();
        expect(catalog.queryByText(PHRASE_DEFINITIONS[0].name)).not.toBeInTheDocument();
        expect(catalog.queryByText('Actions: Say phrase → Exit to root after the phrase action runs')).not.toBeInTheDocument();
    });

    it('expands and collapses the second-apex action with its speech conditions', () => {
        render(<LivePhrases name="live-phrases" />);
        fireEvent.click(screen.getByLabelText('Chicane overtake closure'));
        fireEvent.click(screen.getByLabelText('Chicane overtake entry conditions'));
        const entry = within(screen.getByRole('list', { name: 'Chicane overtake conditions' }));
        expect(entry.getByText('Estimated opponent distance (m) <= 10')).toBeVisible();
        expect(entry.getByText('Opponent estimated time to corner entry (s) <= 2')).toBeVisible();
        const actionSummary = screen.getByLabelText('Chicane overtake action 1: say phrase');
        const sentence = screen.getByText('Go wide in the first turn. then take second apex if possible');
        const speechHeading = within(actionSummary.closest('details')!).getByText('Action conditions');
        expect(sentence).not.toBeVisible();
        expect(speechHeading).not.toBeVisible();
        fireEvent.click(actionSummary);
        expect(sentence).toBeVisible();
        expect(speechHeading).toBeVisible();
        const speechList = screen.getByRole('list', { name: 'Chicane overtake action 1: say phrase conditions' });
        speechList.querySelectorAll('summary').forEach((summary) => fireEvent.click(summary));
        const speech = within(speechList);
        expect(speech.getByText('Opponent estimated time to corner entry (s) < 0.5')).toBeVisible();
        expect(speech.getByText('OR')).toBeVisible();
        expect(speech.getByText('Upcoming corner turns left')).toBeVisible();
        expect(speech.getByText('Opponent near the left edge')).toBeVisible();
        expect(speech.getByText('Upcoming corner turns right')).toBeVisible();
        expect(speech.getByText('Opponent near the right edge')).toBeVisible();
        expect(speech.queryByText(/Player near/)).not.toBeInTheDocument();
        expect(speech.queryByText(/(?:Player|Opponent) in a .*turn corner/)).not.toBeInTheDocument();
        fireEvent.click(actionSummary);
        expect(sentence).not.toBeVisible();
        expect(speechHeading).not.toBeVisible();
        expect(speechList).not.toBeVisible();
        expect(entry.getByText('Estimated opponent distance (m) <= 10')).toBeVisible();
    });

    it('shows the additional chicane speech action with its opposite-side conditions', () => {
        render(<LivePhrases name="live-phrases" />);
        fireEvent.click(screen.getByLabelText('Chicane overtake closure'));
        fireEvent.click(screen.getByLabelText('Chicane overtake action 2: say phrase'));
        expect(screen.getByText('brake early, Hold inside')).toBeVisible();
        const list = screen.getByRole('list', { name: 'Chicane overtake action 2: say phrase conditions' });
        list.querySelectorAll('summary').forEach((summary) => fireEvent.click(summary));
        expect(within(list).getByText('Opponent estimated time to corner entry (s) < 0.5')).toBeVisible();
        expect(within(list).getByText('OR')).toBeVisible();
        expect(within(list).getByText('Upcoming corner turns right')).toBeVisible();
        expect(within(list).getByText('Opponent near the left edge')).toBeVisible();
        expect(within(list).getByText('Upcoming corner turns left')).toBeVisible();
        expect(within(list).getByText('Opponent near the right edge')).toBeVisible();
        expect(screen.getByLabelText('Chicane overtake action 3: exit to root')).toBeVisible();
    });

    it('displays mixed connectors from the evaluated conditions in order', () => {
        const rules = PHRASE_DEFINITIONS.map((rule, index) => index === 0 ? {
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
        fireEvent.click(screen.getByLabelText(`${rules[0].name} closure`));
        fireEvent.click(screen.getByLabelText(`${rules[0].name} entry conditions`));
        const conditions = within(screen.getByRole('list', { name: `${rules[0].name} conditions` }));
        const items = conditions.getAllByRole('listitem');
        expect(items).toHaveLength(3);
        expect(within(items[0]).queryByText(/^(AND|OR)$/)).not.toBeInTheDocument();
        expect(within(items[1]).getByText('OR')).toBeVisible();
        expect(within(items[2]).getByText('AND')).toBeVisible();
        expect(conditions.getAllByText('Missing input')).toHaveLength(3);
        fireEvent.click(screen.getByText('How conditions are evaluated'));
        expect(screen.getByText(/AND is evaluated before OR/)).toBeVisible();
    });

    it('displays nested group boundaries, connectors and each independent status', () => {
        // A OR (B AND (C OR missing)): both groups match while A and the missing leaf do not.
        const rules = PHRASE_DEFINITIONS.map((rule, index) => index === 0 ? {
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
        fireEvent.click(screen.getByLabelText(`${rules[0].name} closure`));
        fireEvent.click(screen.getByLabelText(`${rules[0].name} entry conditions`));
        const list = screen.getByRole('list', { name: `${rules[0].name} conditions` });
        const conditions = within(list);
        const outerGroup = conditions.getByLabelText(`${rules[0].name} conditions group 2`);
        expect(outerGroup).toBeVisible();
        expect(within(outerGroup).getByText('Met')).toBeVisible();
        expect(conditions.getByText('Speed (km/h) >= 30')).not.toBeVisible();
        fireEvent.click(outerGroup);
        expect(conditions.getByText('Speed (km/h) >= 30')).toBeVisible();
        const innerGroup = conditions.getByLabelText('Grouped conditions group 2');
        expect(innerGroup).toBeVisible();
        expect(conditions.getByText('Speed (km/h) <= 110')).not.toBeVisible();
        fireEvent.click(innerGroup);
        const items = conditions.getAllByRole('listitem');
        const groups = conditions.getAllByRole('group', { name: 'Condition group' });
        expect(groups).toHaveLength(2);
        expect(groups[0]).toContainElement(groups[1]);
        expect(items).toHaveLength(6);
        expect(within(items[1]).getAllByText('OR')).toHaveLength(2);
        expect(within(groups[0]).getAllByText('OR')).toHaveLength(2);
        expect(within(items[3]).getByText('AND')).toBeVisible();
        expect(within(groups[1]).getByText('AND')).toBeVisible();
        expect(within(groups[0]).getAllByText('Met')).toHaveLength(4);
        expect(within(groups[1]).getAllByText('Met')).toHaveLength(2);
        for (const item of [items[1], items[3]]) {
            expect(item).toHaveAttribute('data-condition-fit', 'true');
            expect(item).toHaveAttribute('data-input-missing', 'false');
        }
        for (const item of [items[2], items[4]]) {
            expect(within(item).queryByText(/^(AND|OR)$/)).not.toBeInTheDocument();
        }
        const failed = within(items[0]);
        expect(failed.getByText('Speed (km/h) >= 120')).toBeVisible();
        expect(failed.getByText('Not met')).toBeVisible();
        const missing = within(items[5]);
        expect(missing.getByText('Opponent ahead on visible track = 1')).toBeVisible();
        expect(missing.getByText('Missing input')).toBeVisible();
        expect(missing.getByText('OR')).toBeVisible();
        const firstGuide = within(screen.getByRole('region', { name: 'Sentence catalog' })).getAllByRole('listitem')[0];
        expect(within(firstGuide).queryByText(/Waiting for:/)).not.toBeInTheDocument();
        fireEvent.click(outerGroup);
        expect(missing.getByText('Missing input')).not.toBeVisible();
        expect(failed.getByText('Not met')).toBeVisible();
        fireEvent.click(outerGroup);
        expect(missing.getByText('Missing input')).toBeVisible();
        fireEvent.click(screen.getByText('How conditions are evaluated'));
        expect(screen.getByText(/Groups are evaluated first/)).toBeVisible();
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
        const sentence = PHRASE_DEFINITIONS.find((rule) => rule.id === 'inside-outbraking')!.sentence;
        expect(output.getByText(sentence)).toBeInTheDocument();
        expect(output.queryByText('You are accelerating.')).not.toBeInTheDocument();
        expect(screen.getByText('Live Map: Ready')).toBeInTheDocument();
        expect(screen.getByText('Current section: slow corner · entry')).toBeInTheDocument();
        expect(screen.getByText('Corner shape: bend')).toBeInTheDocument();
        fireEvent.click(screen.getByLabelText('Outbraking on the inside closure'));
        fireEvent.click(screen.getByLabelText('Outbraking on the inside entry conditions'));
        const conditions = within(screen.getByRole('list', { name: 'Outbraking on the inside conditions' }));
        const conditionCount = PHRASE_DEFINITIONS.find((rule) => rule.id === 'inside-outbraking')!.conditions.length;
        expect(conditions.getAllByText('Met')).toHaveLength(conditionCount);
        conditions.getAllByRole('listitem').forEach((condition) => expect(condition).toHaveAttribute('data-condition-fit', 'true'));
        act(() => { publish(3, 0); });
        const speedCondition = within(conditions.getByText('Speed (km/h) >= 30').closest('li')!);
        expect(speedCondition.getByText('Not met')).toBeVisible();
        expect(conditions.getAllByText('Met')).toHaveLength(conditionCount - 1);
        fireEvent.click(screen.getByLabelText('Outbraking on the inside closure'));
        expect(speedCondition.getByText('Not met')).not.toBeVisible();
        act(() => { publish(4); });
        const collapsedClosure = screen.getByLabelText('Outbraking on the inside closure');
        expect(collapsedClosure.closest('details')).not.toHaveAttribute('open');
        expect(within(collapsedClosure).getByText('Active')).toBeVisible();
        fireEvent.click(collapsedClosure);
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
        PHRASE_DEFINITIONS.forEach((rule) => expect(catalog.getByLabelText(`${rule.name} closure`)).toBeVisible());
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
