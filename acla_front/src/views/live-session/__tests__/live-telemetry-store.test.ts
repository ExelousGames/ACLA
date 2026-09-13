import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { createLiveTelemetryStore } from '../live-telemetry-store';
import type { RecordingViewUpdate } from '../live-session-types';

const frame = (
    sequence: number,
    sample: RecordingViewUpdate['sample'] = {},
    committedCount = sequence,
): RecordingViewUpdate => ({
    type: 'frame',
    game: 'acc',
    sample: {
        Graphics_status: ACC_STATUS.ACC_LIVE,
        Graphics_sequence: sequence,
        Physics_speed_kmh: sequence,
        ...sample,
    },
    sequence,
    committedSequence: committedCount,
    committedCount,
});

describe('live telemetry store', () => {
    it('delivers 120 frames synchronously, exactly once, and in sequence', () => {
        const store = createLiveTelemetryStore();
        const received: number[] = [];
        store.subscribeEvents((event) => {
            if (event.type === 'frame') received.push(event.update.sequence);
        });

        for (let sequence = 1; sequence <= 120; sequence += 1) {
            expect(store.publishFrame(frame(sequence), { Static_track: 'monza' })).toBe(true);
            const snapshot = store.getSnapshot();
            expect(snapshot.sampleIndex).toBe(sequence - 1);
            expect(snapshot.committedSampleCount).toBe(sequence);
            expect(snapshot.telemetryStatus).toBe(ACC_STATUS.ACC_LIVE);
        }

        expect(received).toEqual(Array.from({ length: 120 }, (_, index) => index + 1));
        expect(store.publishFrame(frame(120))).toBe(false);
        expect(received).toHaveLength(120);
    });

    it('replaces dynamic fields while retaining every first-seen static field', () => {
        const store = createLiveTelemetryStore();
        store.publishFrame(frame(1, {
            Graphics_first_only: 1,
            Physics_first_only: 2,
        }), {
            Static_track: 'monza',
            Static_num_cars: 1,
        });
        store.publishFrame(frame(2, {
            Graphics_second_only: 3,
            Physics_second_only: 4,
        }), {
            Static_track: 'spa',
            Static_num_cars: 99,
            Static_late_key: 'locked-late',
        });

        const snapshot = store.getSnapshot();
        expect(snapshot.graphicsTelemetry).toEqual({
            Graphics_status: ACC_STATUS.ACC_LIVE,
            Graphics_sequence: 2,
            Graphics_second_only: 3,
        });
        expect(snapshot.physicsTelemetry).toEqual({
            Physics_speed_kmh: 2,
            Physics_second_only: 4,
        });
        expect(snapshot.currentTelemetry).toEqual({
            Static_track: 'monza',
            Static_num_cars: 1,
            Static_late_key: 'locked-late',
            ...snapshot.graphicsTelemetry,
            ...snapshot.physicsTelemetry,
        });
        expect(snapshot.currentTelemetry).not.toHaveProperty('Graphics_first_only');
        expect(snapshot.currentTelemetry).not.toHaveProperty('Physics_first_only');
    });

    it('emits explicit stream/session resets and supports count restoration and finalization', () => {
        const store = createLiveTelemetryStore();
        const eventTypes: string[] = [];
        store.subscribeEvents((event) => eventTypes.push(event.type));
        store.publishFrame(frame(1), { Static_track: 'monza' });

        store.beginStream();
        expect(store.getSnapshot()).toMatchObject({
            currentTelemetry: { Static_track: 'monza' },
            sampleIndex: -1,
            telemetryStatus: null,
            committedSampleCount: 1,
            sessionGeneration: 0,
            streamGeneration: 1,
        });
        store.restoreCommittedSampleCount(42);
        expect(store.getSnapshot().committedSampleCount).toBe(42);
        store.finalizeCommittedSampleCount(45);
        expect(store.getSnapshot().committedSampleCount).toBe(45);

        store.resetSession();
        expect(store.getSnapshot()).toMatchObject({
            currentTelemetry: {},
            sampleIndex: -1,
            telemetryStatus: null,
            committedSampleCount: 0,
            sessionGeneration: 1,
            streamGeneration: 2,
        });
        expect(eventTypes).toEqual(['frame', 'stream-reset', 'session-reset']);
    });

    it('replays only the latest frame and isolates listeners and unsubscription mutations', () => {
        const store = createLiveTelemetryStore();
        store.publishFrame(frame(1));
        const received: number[] = [];
        const second = jest.fn();
        let removeSecond: () => void = () => undefined;
        store.subscribeEvents((event) => {
            if (event.type !== 'frame') return;
            received.push(event.update.sequence);
            removeSecond();
            throw new Error('isolated listener failure');
        }, { replayLatest: true });
        removeSecond = store.subscribeEvents(second);

        store.publishFrame(frame(2));
        store.publishFrame(frame(3));

        expect(received).toEqual([1, 2, 3]);
        expect(second).toHaveBeenCalledTimes(1);
        store.beginStream();
        const replayAfterReset = jest.fn();
        store.subscribeEvents(replayAfterReset, true);
        expect(replayAfterReset).not.toHaveBeenCalled();
    });

    it('notifies selector listeners only when their selected value changes', () => {
        const store = createLiveTelemetryStore();
        const counts: number[] = [];
        const remove = store.subscribeSelector(
            (snapshot) => snapshot.committedSampleCount,
            (count) => counts.push(count),
        );

        store.publishFrame(frame(1, {}, 0));
        store.publishFrame(frame(2, {}, 1));
        store.publishFrame(frame(3, {}, 1));
        remove();
        store.publishFrame(frame(4, {}, 2));

        expect(counts).toEqual([1]);
    });
});
