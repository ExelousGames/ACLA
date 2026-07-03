import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import LiveRangeTracker, { LiveRangeTrackerHandle } from '../LiveRangeTracker';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

const sample = (lap: number, position: number) => ({
    Static_track: 'brands_hatch',
    Graphics_completed_laps: lap,
    Graphics_normalized_car_position: position,
});

describe('LiveRangeTracker', () => {
    it('creates one component-owned tracker and replaces the previous tracker', () => {
        const ref = React.createRef<LiveRangeTrackerHandle>();
        render(<LiveRangeTracker ref={ref} sessionMode="live" />);

        act(() => {
            ref.current!.setTracker({
                ranges: [{ id: 'r1', label: 'First', start_position: 0.1, end_position: 0.2 }],
            });
        });
        expect(ref.current!.getTracker().tracker?.ranges).toHaveLength(1);

        act(() => {
            ref.current!.setTracker({
                ranges: [{ id: 'r2', label: 'Second', start_position: 0.3, end_position: 0.4 }],
            });
        });

        const tracker = ref.current!.getTracker().tracker;
        expect(tracker?.ranges).toHaveLength(1);
        expect(tracker?.ranges[0]).toMatchObject({
            id: 'r2',
            label: 'Second',
            lifecycle_status: 'pending',
        });
        expect(screen.getByText('Second')).toBeInTheDocument();
    });

    it('records classifier parent and child segment status', () => {
        const ref = React.createRef<LiveRangeTrackerHandle>();
        render(<LiveRangeTracker ref={ref} sessionMode="live" />);

        act(() => {
            ref.current!.setTracker({
                ranges: [{ id: 'r1', label: 'Tracked range', start_position: 0.1, end_position: 0.2 }],
            });
            ref.current!.updateTracker({
                action: 'record_classification',
                range_id: 'r1',
                classifier_status: 'mistake',
                parent_segment: {
                    labels: ['MSP'],
                    start_index: 10,
                    end_index: 30,
                },
                child_segments: [
                    {
                        labels: ['late_brake'],
                        start_index: 12,
                        end_index: 18,
                    },
                ],
            });
        });

        const range = ref.current!.getTracker().tracker?.ranges[0];
        expect(range).toMatchObject({
            lifecycle_status: 'classified',
            classifier_status: 'mistake',
            parent_segment: {
                labels: ['MSP'],
                start_index: 10,
                end_index: 30,
            },
            child_segments: [
                {
                    labels: ['late_brake'],
                    start_index: 12,
                    end_index: 18,
                },
            ],
        });
        expect(screen.getByText('MSP')).toBeInTheDocument();
        expect(screen.getByText('late_brake')).toBeInTheDocument();
    });

    it('marks a pending range as classifying once the driver crosses the end position', async () => {
        const ref = React.createRef<LiveRangeTrackerHandle>();
        const sessionIntelligence = new SessionIntelligence();
        const sendToolStatus = jest.fn(() => true);
        const first = sample(0, 0.1);
        const middle = sample(0, 0.24);
        const crossed = sample(0, 0.26);

        sessionIntelligence.tick(first);
        const { rerender } = render(
            <LiveRangeTracker
                ref={ref}
                liveData={first}
                sessionMode="live"
                sessionIntelligence={sessionIntelligence}
                sendToolStatus={sendToolStatus}
            />,
        );

        act(() => {
            ref.current!.setTracker({
                ranges: [{ id: 'r1', label: 'Exit range', start_position: 0.1, end_position: 0.25 }],
            });
        });

        sessionIntelligence.tick(middle);
        rerender(
            <LiveRangeTracker
                ref={ref}
                liveData={middle}
                sessionMode="live"
                sessionIntelligence={sessionIntelligence}
                sendToolStatus={sendToolStatus}
            />,
        );

        sessionIntelligence.tick(crossed);
        rerender(
            <LiveRangeTracker
                ref={ref}
                liveData={crossed}
                sessionMode="live"
                sessionIntelligence={sessionIntelligence}
                sendToolStatus={sendToolStatus}
            />,
        );

        await waitFor(() => expect(sendToolStatus).toHaveBeenCalledTimes(1));
        expect(sendToolStatus).toHaveBeenCalledWith(expect.objectContaining({
            event: 'live_range_classification_requested',
            range_id: 'r1',
            telemetry_row_count: 2,
        }));
        expect(ref.current!.getTracker().tracker?.ranges[0]).toMatchObject({
            lifecycle_status: 'classifying',
            start_sample_idx: 0,
            end_sample_idx: 1,
        });
    });

    it('does not trigger a closed tracker', async () => {
        const ref = React.createRef<LiveRangeTrackerHandle>();
        const sessionIntelligence = new SessionIntelligence();
        const sendToolStatus = jest.fn(() => true);
        const first = sample(0, 0.1);
        const crossed = sample(0, 0.3);

        sessionIntelligence.tick(first);
        const { rerender } = render(
            <LiveRangeTracker
                ref={ref}
                liveData={first}
                sessionMode="live"
                sessionIntelligence={sessionIntelligence}
                sendToolStatus={sendToolStatus}
            />,
        );
        act(() => {
            ref.current!.setTracker({
                ranges: [{ id: 'r1', start_position: 0.1, end_position: 0.2 }],
            });
            ref.current!.updateTracker({ action: 'close' });
        });

        sessionIntelligence.tick(crossed);
        rerender(
            <LiveRangeTracker
                ref={ref}
                liveData={crossed}
                sessionMode="live"
                sessionIntelligence={sessionIntelligence}
                sendToolStatus={sendToolStatus}
            />,
        );

        await waitFor(() => expect(sendToolStatus).not.toHaveBeenCalled());
        expect(ref.current!.getTracker().tracker?.ranges[0].lifecycle_status).toBe('pending');
    });
});
