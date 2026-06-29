import { render } from '@testing-library/react';
import {
    BaselineCollectionTracker,
    BASELINE_PROGRESS_MESSAGE_ID,
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
        let messages: any[] = [];

        const props = {
            enabled: true,
            liveData: makeSample(0, 0.001, 10),
            sessionMode: 'live' as const,
            onTagChange: (tag: BaselineCollectionTag | null) => tags.push(tag),
            onLapRecordChange: (record: BaselineLapRecord | null) => records.push(record),
            updateAgentMessages: (updater: (messages: any[]) => any[]) => {
                messages = updater(messages);
            },
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
            ready: true,
            progress_percent: 100,
            status: 'complete',
        });

        rerender(<BaselineCollectionTracker {...props} liveData={null} />);

        const progressMessage = messages.find((message) => message.id === BASELINE_PROGRESS_MESSAGE_ID);
        expect(tags.at(-1)).toMatchObject({
            ready: true,
            progress_percent: 100,
            status: 'complete',
        });
        expect(progressMessage?.progress).toMatchObject({
            value: 100,
            detail: 'Baseline complete. Classifier request is ready.',
        });
        expect(records.filter(Boolean).at(-1)).toBe(completedRecord);
    });

    it('completes the cached baseline when position wraps even before lap counter advances', () => {
        const tags: Array<BaselineCollectionTag | null> = [];
        const records: Array<BaselineLapRecord | null> = [];
        let messages: any[] = [];

        const props = {
            enabled: true,
            liveData: makeSample(3, 0.001, 10),
            sessionMode: 'live' as const,
            onTagChange: (tag: BaselineCollectionTag | null) => tags.push(tag),
            onLapRecordChange: (record: BaselineLapRecord | null) => records.push(record),
            updateAgentMessages: (updater: (messages: any[]) => any[]) => {
                messages = updater(messages);
            },
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
            ready: true,
            progress_percent: 100,
            status: 'complete',
        });
        expect(messages.find((message) => message.id === BASELINE_PROGRESS_MESSAGE_ID)?.progress?.value).toBe(100);
    });
});
