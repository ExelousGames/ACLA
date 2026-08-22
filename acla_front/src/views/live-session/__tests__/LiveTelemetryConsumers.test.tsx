import React from 'react';
import { act, render, screen } from '@testing-library/react';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { LiveSessionContext } from '../LiveSessionContext';
import LiveTelemetryOverview from '../LiveTelemetryOverview';
import LiveTrajectoryMap from '../LiveTrajectoryMap';
import { liveTelemetryStore } from '../live-telemetry-store';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: any) => (
        <div ref={ref} {...props}>{children}</div>
    ));
    const TextFieldRoot = ({ children, ...props }: any) => <div {...props}>{children}</div>;
    return {
        Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
        Box: Div,
        Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
        Card: Div,
        Flex: Div,
        Grid: Div,
        Text: ({ children, ...props }: any) => <span {...props}>{children}</span>,
        TextField: {
            Root: TextFieldRoot,
            Slot: Div,
        },
    };
});

jest.mock('@radix-ui/react-icons', () => ({
    MagnifyingGlassIcon: () => <span>Search</span>,
}));

jest.mock('contexts/CircuitMapsContext', () => ({
    useCircuitMaps: () => ({
        getCircuitMapByTrack: jest.fn(() => new Promise(() => undefined)),
    }),
}));

const publishFrame = (sequence: number) => liveTelemetryStore.publishFrame({
    type: 'frame',
    game: 'acc',
    sample: {
        Graphics_status: ACC_STATUS.ACC_LIVE,
        Graphics_sequence: sequence,
        Graphics_car_coordinates: JSON.stringify([{ x: sequence + 1, y: 1, z: sequence + 2 }]),
        Physics_speed_kmh: sequence,
        Physics_timestamp: sequence / 60,
    },
    sequence,
    committedSequence: sequence,
    committedCount: sequence,
}, { Static_track: 'monza' });

describe('live telemetry latest-value and trajectory consumers', () => {
    beforeEach(() => {
        liveTelemetryStore.resetSession();
        (global as any).ResizeObserver = class {
            observe = jest.fn();
            disconnect = jest.fn();
        };
        HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
            arc: jest.fn(),
            beginPath: jest.fn(),
            clearRect: jest.fn(),
            closePath: jest.fn(),
            createRadialGradient: jest.fn(() => ({ addColorStop: jest.fn() })),
            fill: jest.fn(),
            fillRect: jest.fn(),
            lineTo: jest.fn(),
            moveTo: jest.fn(),
            setTransform: jest.fn(),
            stroke: jest.fn(),
        })) as any;
    });

    it('shows only the newest merged compatibility sample', () => {
        render(<LiveTelemetryOverview name="latest telemetry" />);

        act(() => {
            publishFrame(1);
            publishFrame(2);
        });

        expect(screen.getByText('Static_track')).toBeInTheDocument();
        expect(screen.getByText('monza')).toBeInTheDocument();
        expect(screen.getByText('Graphics_sequence')).toBeInTheDocument();
        expect(screen.getAllByText('2').length).toBeGreaterThan(0);
        expect(screen.queryByText('1')).not.toBeInTheDocument();
    });

    it('retains every one of 120 trajectory frames delivered in one React batch', () => {
        render(
            <LiveSessionContext.Provider value={{ staticData: { Static_track: 'monza' } } as any}>
                <LiveTrajectoryMap name="live trajectory" />
            </LiveSessionContext.Provider>,
        );

        act(() => {
            for (let sequence = 1; sequence <= 120; sequence += 1) publishFrame(sequence);
        });

        expect(screen.getByText('120 visible samples')).toBeInTheDocument();
    });
});
