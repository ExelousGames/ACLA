import React from 'react';
import { act, render, screen } from '@testing-library/react';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    useAiToolComponentRefDirectory,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import { getToolEnvelopeUiOutput } from 'views/lap-analysis/ai-chat/ai-tool-base';
import BaselineCollection, { type BaselineCollectionHandle } from '../BaselineCollection';
import { LiveSessionContext } from '../LiveSessionContext';

const makeSample = (lap: number, position: number, currentTime: number) => ({
    Static_track: 'brands_hatch',
    Static_car_model: 'Ferrari 296',
    Static_num_cars: 1,
    Graphics_completed_laps: lap,
    Graphics_normalized_car_position: position,
    Graphics_current_time: currentTime,
});

let directory: AiToolComponentRefDirectory | null = null;

const DirectoryObserver = () => {
    directory = useAiToolComponentRefDirectory();
    return null;
};

const Harness = ({
    telemetry,
    show = true,
}: {
    telemetry: Record<string, any>;
    show?: boolean;
}) => (
    <AiToolComponentRefProvider>
        <DirectoryObserver />
        <LiveSessionContext.Provider value={{ currentTelemetry: telemetry } as any}>
            {show && <BaselineCollection name={AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION} />}
        </LiveSessionContext.Provider>
    </AiToolComponentRefProvider>
);

const getHandle = () => directory!
    .findComponentRef<BaselineCollectionHandle>(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)!
    .current!;

describe('BaselineCollection visualization', () => {
    beforeEach(() => {
        directory = null;
    });

    it('waits for the next boundary, records one lap, reports progress, and emits completion once', () => {
        const outputs: any[] = [];
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = getHandle();
        handle.subscribeToolOutput((output) => outputs.push(output));

        act(() => {
            expect(handle.startCollection()).toMatchObject({
                status: 'waiting_for_start',
                progress_percent: 0,
            });
        });
        expect(handle.getLapRecord()).toBeNull();

        view.rerender(<Harness telemetry={makeSample(4, 0.9, 90_000)} />);
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start', progress_percent: 0 });

        view.rerender(<Harness telemetry={makeSample(5, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(5, 0.4, 40_000)} />);
        expect(handle.getTag()).toMatchObject({ status: 'collecting', progress_percent: 40, baseline_lap: 5 });

        view.rerender(<Harness telemetry={makeSample(5, 0.98, 98_000)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.2, 20_000)} />);

        expect(handle.getLapRecord()).toMatchObject({
            lap: 5,
            track: 'brands_hatch',
            car: 'Ferrari 296',
            sample_count: 3,
        });
        expect(handle.getLapRecord()?.records.map((row) => row.Graphics_normalized_car_position))
            .toEqual([0.001, 0.4, 0.98]);
        expect(handle.getTag()).toMatchObject({ status: 'complete', progress_percent: 100 });
        expect(outputs).toHaveLength(1);
        expect(getToolEnvelopeUiOutput(outputs[0])).toEqual({
            progress_percent: 100,
            status: 'complete',
            car: 'Ferrari 296',
            track: 'brands_hatch',
            message: 'Baseline complete. Cached lap record is ready.',
        });
        expect(handle.getToolOutput()).toBe(outputs[0]);
    });

    it('completes when position wraps before the lap counter advances', () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        act(() => { handle.startCollection(); });

        view.rerender(<Harness telemetry={makeSample(3, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.5, 50_000)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.99, 99_000)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.002, 20)} />);

        expect(handle.getLapRecord()).toMatchObject({ lap: 3, sample_count: 3 });
        expect(handle.getTag()).toMatchObject({ status: 'complete', progress_percent: 100 });
    });

    it('restarts the mounted collector and waits for a fresh lap start', () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        act(() => { handle.startCollection(); });
        view.rerender(<Harness telemetry={makeSample(0, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(0, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.001, 5)} />);
        expect(handle.getLapRecord()).not.toBeNull();

        act(() => { handle.restartCollection(); });
        expect(handle.getLapRecord()).toBeNull();
        expect(handle.getToolOutput()).toBeNull();
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start', baseline_lap: null });

        view.rerender(<Harness telemetry={makeSample(1, 0.002, 20)} />);
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start' });
        view.rerender(<Harness telemetry={makeSample(1, 0.4, 40_000)} />);
        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} />);
        expect(handle.getTag()).toMatchObject({ status: 'collecting', baseline_lap: 2 });
    });

    it('unregisters and discards partial or completed state when closed, then reopens fresh', () => {
        const view = render(<Harness telemetry={{}} />);
        const firstHandle = getHandle();
        act(() => { firstHandle.startCollection(); });
        view.rerender(<Harness telemetry={makeSample(0, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(0, 0.5, 50_000)} />);
        expect(firstHandle.getTag()).toMatchObject({ status: 'collecting' });

        view.rerender(<Harness telemetry={makeSample(0, 0.5, 50_000)} show={false} />);
        expect(directory!.findComponentRef(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)).toBeNull();
        expect(firstHandle.getTag()).toBeNull();
        expect(firstHandle.getLapRecord()).toBeNull();
        expect(firstHandle.getToolOutput()).toBeNull();

        view.rerender(<Harness telemetry={makeSample(0, 0.5, 50_000)} />);
        const secondHandle = getHandle();
        expect(secondHandle).not.toBe(firstHandle);
        expect(secondHandle.getTag()).toBeNull();
        act(() => { secondHandle.startCollection(); });
        expect(secondHandle.getTag()).toMatchObject({ status: 'waiting_for_start', baseline_lap: null });
        expect(screen.getByLabelText('Baseline collection progress')).toBeInTheDocument();

        view.rerender(<Harness telemetry={makeSample(0, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} />);
        expect(secondHandle.getLapRecord()).not.toBeNull();

        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} show={false} />);
        expect(secondHandle.getTag()).toBeNull();
        expect(secondHandle.getLapRecord()).toBeNull();
        expect(secondHandle.getToolOutput()).toBeNull();
    });
});
