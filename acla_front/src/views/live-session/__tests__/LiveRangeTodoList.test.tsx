import {
    LiveRangeTodoListRunner,
    calculateForwardCircularDistance,
    calculateLiveRangeEta,
    crossedLiveRangeTodoPosition,
} from 'components/ai-engineering-tools/LiveRangeTodoList';
import type { LiveRangeTodoEventInput } from 'components/ai-engineering-tools';

const event = (id: string, normalizedPosition: number): LiveRangeTodoEventInput => ({
    id,
    normalized_position: normalizedPosition,
    lead_time_seconds: 0,
    content: { title: id },
    data: { event: `${id}_due` },
});

describe('live range helpers', () => {
    it('calculates circular distance, ETA, and rollover crossings', () => {
        expect(calculateForwardCircularDistance(0.9, 0.1)).toBeCloseTo(0.2);
        expect(calculateLiveRangeEta(0.9, 0.1, 0.1)).toBeCloseTo(2);
        expect(crossedLiveRangeTodoPosition(
            { position: 0.95, receivedAt: 0, lap: 2 },
            { position: 0.05, receivedAt: 1000, lap: 3 },
            0.99,
        )).toBe(true);
    });
});

describe('LiveRangeTodoListRunner promise ownership', () => {
    it('keeps an invocation result pending until every event it created becomes due', async () => {
        const runner = new LiveRangeTodoListRunner('live-range');
        const result = runner.replaceEvents([event('one', 0.2), event('two', 0.4)]);
        const operation = runner.createOwnedOperation(result, ['one', 'two']);
        let settled = false;
        void operation.result.then(() => { settled = true; });

        runner.acceptTelemetry({ Graphics_normalized_car_position: 0, Graphics_completed_laps: 1 });
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.25, Graphics_completed_laps: 1 });
        await operation.statuses[0];
        expect(settled).toBe(false);

        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.45, Graphics_completed_laps: 1 });

        await expect(Promise.all(operation.statuses)).resolves.toEqual([
            expect.objectContaining({ event_id: 'one', event: 'one_due' }),
            expect.objectContaining({ event_id: 'two', event: 'two_due' }),
        ]);
        await expect(operation.result).resolves.toMatchObject({
            status: 'empty',
            event_count: 0,
        });
    });

    it('rejects a displaced operation while a replacement operation owns only re-armed events', async () => {
        const runner = new LiveRangeTodoListRunner('live-range');
        const original = runner.createOwnedOperation(
            runner.replaceEvents([event('one', 0.2), event('two', 0.4)]),
            ['one', 'two'],
        );

        const updated = runner.updateEvents([{ id: 'one', normalized_position: 0.3 }]);
        const replacement = runner.createOwnedOperation(updated, ['one']);

        await expect(original.result).rejects.toMatchObject({
            name: 'InvalidLiveRangeTodoListError',
        });
        expect(replacement.statuses).toHaveLength(1);
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0, Graphics_completed_laps: 1 });
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.35, Graphics_completed_laps: 1 });
        await expect(replacement.result).resolves.toMatchObject({ status: 'ready', event_count: 1 });
    });

    it('settles all outstanding promises when cleared or disposed', async () => {
        const runner = new LiveRangeTodoListRunner('live-range');
        const cleared = runner.createOwnedOperation(
            runner.replaceEvents([event('one', 0.8)]),
            ['one'],
        );
        runner.clear();
        await expect(cleared.result).rejects.toBeInstanceOf(Error);
        await expect(Promise.allSettled(cleared.statuses)).resolves.toEqual([
            expect.objectContaining({ status: 'rejected' }),
        ]);

        const disposed = runner.createOwnedOperation(
            runner.replaceEvents([event('two', 0.9)]),
            ['two'],
        );
        runner.dispose();
        await expect(disposed.result).rejects.toBeInstanceOf(Error);
        await expect(Promise.allSettled(disposed.statuses)).resolves.toEqual([
            expect.objectContaining({ status: 'rejected' }),
        ]);
    });
});
