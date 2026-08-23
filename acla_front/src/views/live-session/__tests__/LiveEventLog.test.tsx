import React, { useLayoutEffect } from 'react';
import { render, screen } from '@testing-library/react';
import LiveEventLog, { LiveEventLogHandle } from '../LiveEventLog';
import { liveTelemetryStore } from '../live-telemetry-store';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = ({ children, ...props }: any) => <div {...props}>{children}</div>;
    return {
        Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
        Box: Div,
        Flex: Div,
        Table: {
            Root: ({ children }: any) => <table>{children}</table>,
            Header: ({ children }: any) => <thead>{children}</thead>,
            Body: ({ children }: any) => <tbody>{children}</tbody>,
            Row: ({ children }: any) => <tr>{children}</tr>,
            ColumnHeaderCell: ({ children }: any) => <th>{children}</th>,
            Cell: ({ children }: any) => <td>{children}</td>,
        },
        Text: ({ children }: any) => <span>{children}</span>,
        TextField: {
            Root: Div,
            Slot: Div,
        },
    };
});

jest.mock('@radix-ui/react-icons', () => ({
    MagnifyingGlassIcon: () => <span>Search</span>,
}));

const telemetry = (speed: number) => ({
    Static_track: 'brands_hatch',
    Graphics_completed_lap: 2,
    Graphics_normalized_car_position: 0.4,
    Physics_speed_kmh: speed,
});

const Harness = ({
    open,
    speed,
    sampleIndex,
    eventLogRef,
}: {
    open: boolean;
    speed: number;
    sampleIndex: number;
    eventLogRef: React.RefObject<LiveEventLogHandle | null>;
}) => {
    useLayoutEffect(() => {
        liveTelemetryStore.publishFrame({
            type: 'frame',
            game: 'acc',
            sample: telemetry(speed),
            sequence: sampleIndex + 1,
            committedSequence: sampleIndex + 1,
            committedCount: sampleIndex + 1,
        }, { Static_track: 'brands_hatch' });
    }, [sampleIndex, speed]);
    return open ? <LiveEventLog ref={eventLogRef} name="visualization:event-log" /> : null;
};

describe('LiveEventLog telemetry ownership', () => {
    beforeEach(() => liveTelemetryStore.resetSession());

    it('tracks context telemetry only while the visualization is mounted', () => {
        const eventLogRef = React.createRef<LiveEventLogHandle>();
        const { rerender } = render(
            <Harness open speed={120} sampleIndex={10} eventLogRef={eventLogRef} />,
        );

        expect(screen.getByText('0 detected events')).toBeInTheDocument();

        rerender(<Harness open speed={50} sampleIndex={11} eventLogRef={eventLogRef} />);
        expect(screen.getByText('1 detected events')).toBeInTheDocument();
        expect(screen.getByText('CRASHED')).toBeInTheDocument();
        expect(eventLogRef.current?.getAllEvents()).toHaveLength(1);

        rerender(<Harness open={false} speed={120} sampleIndex={12} eventLogRef={eventLogRef} />);
        expect(screen.queryByText(/detected events/)).not.toBeInTheDocument();

        rerender(<Harness open speed={50} sampleIndex={13} eventLogRef={eventLogRef} />);
        expect(screen.getByText('0 detected events')).toBeInTheDocument();
        expect(screen.queryByText('CRASHED')).not.toBeInTheDocument();
        expect(eventLogRef.current?.getAllEvents()).toEqual([]);
    });
});
