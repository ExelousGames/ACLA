import { render } from '@testing-library/react';
import {
    BaselineCollectionTracker,
    type BaselineCollectionTag,
    type BaselineLapRecord,
} from '../BaselineCollectionTracker';

const makeSample = (lap: number, position: number, currentTime: number) => ({
    Static_track: 'brands_hatch',
    Static_car_model: 'Ferrari 296',
    Static_num_cars: 1,
    Graphics_completed_laps: lap,
    Graphics_normalized_car_position: position,
    Graphics_current_time: currentTime,
});

describe('BaselineCollectionTracker', () => {
    it('keeps completed baseline progress and cached rows when live telemetry pauses', () => {
        const tags: Array<BaselineCollectionTag | null> = [];
        const records: Array<BaselineLapRecord | null> = [];

        const props = {
            enabled: true,
            liveData: makeSample(0, 0.001, 10),
            sessionMode: 'live' as const,
            onTagChange: (tag: BaselineCollectionTag | null) => tags.push(tag),
            onLapRecordChange: (record: BaselineLapRecord | null) => records.push(record),
        };

        const { rerender } = render(<BaselineCollectionTracker {...props} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(0, 0.4, 40000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(0, 0.98, 98000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(1, 0.001, 5)} />);

        const completedRecord = records.filter(Boolean).at(-1);
        expect(completedRecord).toMatchObject({
            lap: 0,
            track: 'brands_hatch',
            car: 'Ferrari 296',
            sample_count: 3,
        });
        expect(completedRecord?.records.map((row) => row.Graphics_normalized_car_position)).toEqual([
            0.001,
            0.4,
            0.98,
        ]);
        expect(tags.at(-1)).toMatchObject({
            progress_percent: 100,
            status: 'complete',
            car: 'Ferrari 296',
            track: 'brands_hatch',
        });
        expect(tags.at(-1)).not.toHaveProperty('ready');
        expect(tags.at(-1)).not.toHaveProperty('snapshot');

        rerender(<BaselineCollectionTracker {...props} liveData={null} />);

        expect(tags.at(-1)).toMatchObject({
            progress_percent: 100,
            status: 'complete',
            car: 'Ferrari 296',
            track: 'brands_hatch',
        });
        expect(tags.at(-1)).not.toHaveProperty('ready');
        expect(tags.at(-1)).not.toHaveProperty('snapshot');
        expect(records.filter(Boolean).at(-1)).toBe(completedRecord);
    });

    it('completes the cached baseline when position wraps even before lap counter advances', () => {
        const tags: Array<BaselineCollectionTag | null> = [];
        const records: Array<BaselineLapRecord | null> = [];

        const props = {
            enabled: true,
            liveData: makeSample(3, 0.001, 10),
            sessionMode: 'live' as const,
            onTagChange: (tag: BaselineCollectionTag | null) => tags.push(tag),
            onLapRecordChange: (record: BaselineLapRecord | null) => records.push(record),
        };

        const { rerender } = render(<BaselineCollectionTracker {...props} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(3, 0.5, 50000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(3, 0.99, 99000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(3, 0.002, 20)} />);

        expect(records.filter(Boolean).at(-1)).toMatchObject({
            lap: 3,
            sample_count: 3,
        });
        expect(tags.at(-1)).toMatchObject({
            progress_percent: 100,
            status: 'complete',
            car: 'Ferrari 296',
            track: 'brands_hatch',
        });
        expect(tags.at(-1)).not.toHaveProperty('ready');
        expect(tags.at(-1)).not.toHaveProperty('snapshot');
    });

    it('emits a compact public tool result when the baseline completes', () => {
        const outputs: any[] = [];
        const props = {
            enabled: true,
            liveData: makeSample(0, 0.001, 10),
            sessionMode: 'live' as const,
            onTagChange: jest.fn(),
            onLapRecordChange: jest.fn(),
            onToolOutput: (output: any) => outputs.push(output),
        };

        const { rerender } = render(<BaselineCollectionTracker {...props} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(0, 0.4, 40000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(0, 0.98, 98000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(1, 0.001, 5)} />);

        expect(outputs.at(-1)).toMatchObject({
            status: 'complete',
            progress_percent: 100,
            message: 'Baseline complete. Cached lap record is ready.',
            tool_name: 'collect_live_baseline',
            final: true,
            payload: {
                progress_percent: 100,
                status: 'complete',
                car: 'Ferrari 296',
                track: 'brands_hatch',
                message: 'Baseline complete. Cached lap record is ready.',
            },
        });
        expect(Object.keys(outputs.at(-1).payload).sort()).toEqual([
            'car',
            'message',
            'progress_percent',
            'status',
            'track',
        ]);
        expect(outputs.at(-1)).not.toHaveProperty('baseline');
        expect(outputs.at(-1)).not.toHaveProperty('snapshot');
        expect(outputs.at(-1)).not.toHaveProperty('source');
    });

    it('clears completed baseline state and emits a fresh tool result after restart', () => {
        const tags: Array<BaselineCollectionTag | null> = [];
        const records: Array<BaselineLapRecord | null> = [];
        const outputs: any[] = [];
        const props = {
            enabled: true,
            restartToken: 0,
            liveData: makeSample(0, 0.001, 10),
            sessionMode: 'live' as const,
            onTagChange: (tag: BaselineCollectionTag | null) => tags.push(tag),
            onLapRecordChange: (record: BaselineLapRecord | null) => records.push(record),
            onToolOutput: (output: any) => outputs.push(output),
        };

        const { rerender } = render(<BaselineCollectionTracker {...props} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(0, 0.4, 40000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(0, 0.98, 98000)} />);
        rerender(<BaselineCollectionTracker {...props} liveData={makeSample(1, 0.001, 5)} />);

        const firstRunId = outputs.at(-1).run_id;
        expect(outputs).toHaveLength(1);
        expect(records.filter(Boolean).at(-1)).toMatchObject({ lap: 0 });

        rerender(
            <BaselineCollectionTracker
                {...props}
                restartToken={1}
                liveData={null}
            />
        );

        expect(outputs).toHaveLength(1);
        expect(records.at(-1)).toBeNull();
        expect(tags.some((tag) => tag === null)).toBe(true);

        rerender(
            <BaselineCollectionTracker
                {...props}
                restartToken={1}
                liveData={makeSample(2, 0.001, 10)}
            />
        );
        rerender(
            <BaselineCollectionTracker
                {...props}
                restartToken={1}
                liveData={makeSample(2, 0.45, 45000)}
            />
        );
        rerender(
            <BaselineCollectionTracker
                {...props}
                restartToken={1}
                liveData={makeSample(2, 0.99, 99000)}
            />
        );
        rerender(
            <BaselineCollectionTracker
                {...props}
                restartToken={1}
                liveData={makeSample(3, 0.001, 5)}
            />
        );

        expect(outputs).toHaveLength(2);
        expect(outputs.at(-1).run_id).not.toBe(firstRunId);
        expect(records.filter(Boolean).at(-1)).toMatchObject({ lap: 2 });
    });
});
