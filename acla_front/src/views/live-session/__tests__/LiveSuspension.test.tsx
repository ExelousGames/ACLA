import React from 'react';
import { act, render, screen, within } from '@testing-library/react';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import LiveSuspension from '../LiveSuspension';
import { liveTelemetryStore } from '../live-telemetry-store';
import type { LiveTelemetry } from '../live-session-types';

const publishFrame = (sequence: number, sample: LiveTelemetry) => liveTelemetryStore.publishFrame({
    type: 'frame', game: 'acc', sample, sequence,
    committedSequence: sequence, committedCount: sequence,
});

const reading = (wheel: string) => within(screen.getByRole('group', { name: `${wheel} suspension travel` }));

describe('LiveSuspension', () => {
    beforeEach(() => { liveTelemetryStore.resetSession(); });
    afterEach(() => { act(() => { liveTelemetryStore.resetSession(); }); });

    it('maps the latest four corners to mm and clears readings when the stream or session resets', () => {
        render(<LiveSuspension name="visualization:suspension" />);
        expect(screen.getByText('Waiting for suspension data')).toBeInTheDocument();
        expect(screen.getAllByText('No data')).toHaveLength(4);

        act(() => {
            publishFrame(1, { Physics_suspension_travel_front_left: 0.01 });
            publishFrame(2, {
                Graphics_status: ACC_STATUS.ACC_LIVE,
                Physics_suspension_travel_front_left: 0.034,
                Physics_suspension_travel_front_right: 0.031,
                Physics_suspension_travel_rear_left: 0.039,
                Physics_suspension_travel_rear_right: 0.037,
            });
        });
        expect(reading('Front left').getByText('34.0')).toBeInTheDocument();
        expect(reading('Front right').getByText('31.0')).toBeInTheDocument();
        expect(reading('Rear left').getByText('39.0')).toBeInTheDocument();
        expect(reading('Rear right').getByText('37.0')).toBeInTheDocument();
        expect(screen.getByText('Live telemetry')).toBeInTheDocument();

        act(() => { liveTelemetryStore.beginStream(); });
        expect(screen.getAllByText('No data')).toHaveLength(4);
        act(() => { publishFrame(1, { Physics_suspension_travel_front_left: 0.012 }); });
        expect(reading('Front left').getByText('12.0')).toBeInTheDocument();
        act(() => { liveTelemetryStore.resetSession(); });
        expect(screen.getAllByText('No data')).toHaveLength(4);
    });

    it('preserves zero, signed and large readings on a shared scale and does not retain absent corners', () => {
        render(<LiveSuspension name="visualization:suspension" />);
        act(() => {
            publishFrame(1, {
                Physics_suspension_travel_front_left: 0,
                Physics_suspension_travel_front_right: -0.0125,
                Physics_suspension_travel_rear_left: 0.175,
            });
        });
        expect(reading('Front left').getByText('0.0')).toBeInTheDocument();
        expect(reading('Front right').getByText('-12.5')).toBeInTheDocument();
        expect(reading('Rear left').getByText('175.0')).toBeInTheDocument();
        expect(reading('Rear right').getByText('No data')).toBeInTheDocument();
        expect(screen.getByText('Shared scale: -50 to 200 mm')).toBeInTheDocument();
        expect(screen.getByText('3/4 wheels available')).toBeInTheDocument();

        act(() => { publishFrame(2, { Physics_suspension_travel_rear_right: 0.02 }); });
        expect(reading('Rear right').getByText('20.0')).toBeInTheDocument();
        expect(reading('Front left').getByText('No data')).toBeInTheDocument();
        expect(screen.getByText('1/4 wheels available')).toBeInTheDocument();
    });

    it('shows invalid snapshot values as unavailable and distinguishes paused and replay telemetry', () => {
        const { rerender } = render(<LiveSuspension name="visualization:suspension" telemetry={{
            Physics_suspension_travel_front_left: NaN,
            Physics_suspension_travel_front_right: Infinity,
            Physics_suspension_travel_rear_left: null as any,
            Physics_suspension_travel_rear_right: '0.034' as any,
        }} />);
        expect(screen.getAllByText('No data')).toHaveLength(4);
        rerender(<LiveSuspension name="visualization:suspension" telemetry={{
            Graphics_status: ACC_STATUS.ACC_PAUSE,
            Physics_suspension_travel_front_left: 0.034,
        }} />);
        expect(screen.getByText('Paused')).toBeInTheDocument();
        expect(reading('Front left').getByText('34.0')).toBeInTheDocument();
        rerender(<LiveSuspension name="visualization:suspension" telemetry={{
            Graphics_status: ACC_STATUS.ACC_REPLAY,
            Physics_suspension_travel_front_left: 0.031,
        }} />);
        expect(screen.getByText('Replay')).toBeInTheDocument();
        expect(reading('Front left').getByText('31.0')).toBeInTheDocument();
    });
});
