import React from 'react';
import { act, render } from '@testing-library/react';
import LiveTrajectoryMap, { LiveTrajectoryMapHandle } from '../LiveTrajectoryMap';
import { circuitMap } from '../live-phrases/test-fixtures';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
import { useLiveTelemetrySelector } from '../live-telemetry-store';

jest.mock('radix-ui/internal', () => jest.requireActual('radix-ui/dist/internal.js'), { virtual: true });
jest.mock('contexts/CircuitMapsContext', () => ({ useCircuitMaps: jest.fn() }));
jest.mock('../live-telemetry-store', () => ({ useLiveTelemetrySelector: jest.fn() }));
jest.mock('../LiveSessionContext', () => ({
    LiveSessionContext: require('react').createContext({ sessionGame: null, staticData: {} }),
}));

it('publishes the loaded map and clears it while a different circuit loads', async () => {
    const originalObserver = window.ResizeObserver;
    window.ResizeObserver = jest.fn(() => ({ observe: jest.fn(), disconnect: jest.fn() })) as any;
    const canvas = jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const first = circuitMap();
    const second = { ...circuitMap('fast'), id: 'second', source_track_key: 'spa' };
    let resolveFirst!: (map: typeof first) => void;
    let resolveSecond!: (map: typeof first) => void;
    const lookup = jest.fn()
        .mockReturnValueOnce(new Promise((resolve) => { resolveFirst = resolve; }))
        .mockReturnValueOnce(new Promise((resolve) => { resolveSecond = resolve; }));
    (useCircuitMaps as jest.Mock).mockReturnValue({ getCircuitMapByTrack: lookup });
    const telemetry = { currentTelemetry: { Static_track: 'test' }, telemetryStatus: 2, game: 'acc' };
    (useLiveTelemetrySelector as jest.Mock).mockImplementation(() => telemetry);
    const ref = React.createRef<LiveTrajectoryMapHandle>();
    const view = render(<LiveTrajectoryMap name="visualization:live-trajectory-map" ref={ref} />);
    try {
        const listener = jest.fn();
        const unsubscribe = ref.current!.subscribeCircuitMap(listener);
        expect(ref.current!.getCircuitMap()).toBeNull();
        await act(async () => { resolveFirst(first); });
        expect(ref.current!.getCircuitMap()).toBe(first);
        expect(listener).toHaveBeenCalledTimes(1);

        telemetry.currentTelemetry.Static_track = 'spa';
        view.rerender(<LiveTrajectoryMap name="visualization:live-trajectory-map" ref={ref} />);
        expect(ref.current!.getCircuitMap()).toBeNull();
        expect(listener).toHaveBeenCalledTimes(2);
        await act(async () => { resolveSecond(second); });
        expect(ref.current!.getCircuitMap()).toBe(second);
        expect(listener).toHaveBeenCalledTimes(3);
        expect(lookup).toHaveBeenNthCalledWith(2, 'acc', 'spa');
        unsubscribe();
        telemetry.currentTelemetry.Static_track = '';
        view.rerender(<LiveTrajectoryMap name="visualization:live-trajectory-map" ref={ref} />);
        expect(ref.current!.getCircuitMap()).toBeNull();
        expect(listener).toHaveBeenCalledTimes(3);
    } finally {
        view.unmount();
        canvas.mockRestore();
        window.ResizeObserver = originalObserver;
    }
});
