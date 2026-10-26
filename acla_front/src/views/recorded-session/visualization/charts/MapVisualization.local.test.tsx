import { useContext } from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Theme } from '@radix-ui/themes';
import apiService from 'services/api.service';
import type { RecordedSessionSource } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../../analysis-context';
import { RecordedSessionDataProvider } from '../../data/RecordedSessionDataProvider';
import MapVisualization from './MapVisualization';

jest.mock('radix-ui/internal', () => jest.requireActual('radix-ui/dist/internal.js'), { virtual: true });
jest.mock('services/api.service', () => ({ __esModule: true, default: { post: jest.fn() } }));
const mockGetCircuitMap = jest.fn().mockResolvedValue(null);
const mockGetLabelName = jest.fn();
jest.mock('contexts/CircuitMapsContext', () => ({ useCircuitMaps: () => ({ getCircuitMapByTrack: mockGetCircuitMap }) }));
jest.mock('contexts/AiLabelsContext', () => ({ useAiLabels: () => ({ getLabelName: mockGetLabelName }) }));

const playbackSummary = jest.fn();
const runAnalysis = jest.fn().mockResolvedValue({ status: 'ready' });
const rows = [0, 1].map((index) => ({
    Graphics_current_time: index * 1000,
    Graphics_player_car_id: 0,
    Graphics_car_id: [0],
    Graphics_car_coordinates: [{ x: 100 + index, y: 1, z: 50 + index }],
    Physics_speed_kmh: 180 + index,
}));

const Harness = ({ local = true, telemetry = rows, selected = true, game }: {
    local?: boolean;
    telemetry?: Record<string, any>[];
    selected?: boolean;
    game?: RecordedSessionSource;
}) => {
    const defaults = useContext(AnalysisContext);
    const session = selected ? {
        storage: local ? 'local' as const : 'cloud' as const,
        game_recorded_from: game,
        SessionId: local ? 'local-ibt:1' : 'cloud-1', session_name: 'spa.ibt',
        map: 'spa', car: 'GT3', user_id: '', points: [], data: local ? telemetry : [],
    } : null;
    return <Theme><AnalysisContext.Provider value={{
        ...defaults,
        setRecordedPlaybackSummary: playbackSummary,
        runRecordedAiAnalysis: runAnalysis,
        sessionSelected: session,
    }}><RecordedSessionDataProvider session={session} map="spa">
        <MapVisualization id="local-map" name="local-map" />
    </RecordedSessionDataProvider></AnalysisContext.Provider></Theme>;
};

it('plays local telemetry without cloud requests and enables the shared analysis action', async () => {
    const originalObserver = global.ResizeObserver;
    global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as any;
    const canvas = jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    try {
        render(<Harness />);
        expect(screen.getByText('2 samples')).toBeInTheDocument();
        expect(playbackSummary).toHaveBeenLastCalledWith(expect.objectContaining({ sessionId: 'local-ibt:1', sampleCount: 2, durationSeconds: 1 }));
        expect(apiService.post).not.toHaveBeenCalled();
        expect(mockGetCircuitMap).not.toHaveBeenCalled();
        const analyze = screen.getByRole('button', { name: 'Run AI Analysis' });
        expect(analyze).toBeEnabled();
        await act(async () => { fireEvent.click(analyze); });
        expect(runAnalysis).toHaveBeenCalledTimes(1);
    } finally {
        canvas.mockRestore();
        global.ResizeObserver = originalObserver;
    }
});

it('plays cloud rows from the shared table when selected-session metadata has no data', async () => {
    const originalObserver = global.ResizeObserver;
    global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as any;
    const canvas = jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    (apiService.post as jest.Mock).mockReset()
        .mockResolvedValueOnce({ data: { downloadId: 'cloud-download', sessionMetadata: [{ sessionId: 'cloud-1', chunkCount: 1 }] } })
        .mockResolvedValueOnce({ data: [{ extra: 'preserved but not drawable' }, ...rows] });
    try {
        render(<Harness local={false} />);
        expect(screen.getByRole('button', { name: 'Run AI Analysis' })).toBeDisabled();
        expect(await screen.findByText('2 samples')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Run AI Analysis' })).toBeEnabled();
        await waitFor(() => expect(playbackSummary).toHaveBeenLastCalledWith(expect.objectContaining({
            sessionId: 'cloud-1', sampleCount: 2, durationSeconds: 1,
        })));
        expect(apiService.post).toHaveBeenCalledTimes(2);
        const overview = within(screen.getByRole('region', { name: 'Telemetry Overview' }));
        expect(overview.getByText('181')).toBeInTheDocument();
        expect(overview.getByText(/Sample 3 of 3/)).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Restart playback' }));
        expect(overview.getByText('180')).toBeInTheDocument();
        expect(overview.getByText(/Sample 2 of 3/)).toBeInTheDocument();
    } finally {
        canvas.mockRestore();
        global.ResizeObserver = originalObserver;
    }
});

describe('recorded map coordinate planes', () => {
    const originalObserver = global.ResizeObserver;

    afterEach(() => {
        jest.restoreAllMocks();
        global.ResizeObserver = originalObserver;
    });

    it.each([
        ['acc', false, 'y', 'z'],
        ['iracing', false, 'z', 'y'],
        ['iracing_live', false, 'z', 'y'],
        ['iracing_recorded', true, 'z', 'y'],
        [undefined, false, 'y', 'z'],
        [undefined, true, 'z', 'y'],
    ] as const)('projects %s (local=%s) on X/%s with altitude on %s', async (game, local, verticalAxis, depthAxis) => {
        mockGetCircuitMap.mockClear();
        global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as any;
        const context = {
            setTransform: jest.fn(), clearRect: jest.fn(), fillRect: jest.fn(),
            createLinearGradient: jest.fn(() => ({ addColorStop: jest.fn() })),
            save: jest.fn(), restore: jest.fn(), beginPath: jest.fn(), closePath: jest.fn(),
            moveTo: jest.fn(), lineTo: jest.fn(), stroke: jest.fn(), fill: jest.fn(),
            setLineDash: jest.fn(), translate: jest.fn(), arc: jest.fn(), fillText: jest.fn(),
        };
        jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as any);
        // Both cars share X and altitude, so only the ground-plane axis separates them.
        const position = (vertical: number, altitude = 7) => ({
            x: 10, y: 0, z: 0, [verticalAxis]: vertical, [depthAxis]: altitude,
        });
        const telemetry = [{
            ...rows[0], Graphics_car_id: [0, 1],
            Graphics_car_coordinates: [position(0), position(20)],
        }];
        (apiService.post as jest.Mock).mockReset()
            .mockResolvedValueOnce({ data: { downloadId: 'axes', sessionMetadata: [{ sessionId: 'cloud-1', chunkCount: 1 }] } })
            .mockResolvedValueOnce({ data: telemetry });
        render(<Harness local={local} game={game} telemetry={telemetry} />);
        await screen.findByText('1 samples');
        if (verticalAxis === 'z') expect(mockGetCircuitMap).not.toHaveBeenCalled();
        else expect(mockGetCircuitMap).toHaveBeenCalledWith('acc', 'spa');

        const markerPositions = () => context.translate.mock.calls.slice(-2);
        const originalPositions = markerPositions();
        expect(originalPositions).toHaveLength(2);
        expect(originalPositions[0][0]).toBe(originalPositions[1][0]);
        expect(originalPositions[0][1]).not.toBe(originalPositions[1][1]);
        // Grid lines must span both screen dimensions on the same plane as the cars.
        const gridStart = context.moveTo.mock.calls.slice(-22);
        const gridEnd = context.lineTo.mock.calls.slice(-22);
        expect(gridStart[0][0]).toBeCloseTo(gridEnd[0][0]);
        expect(gridStart[0][1]).not.toBeCloseTo(gridEnd[0][1]);
        expect(gridStart[1][0]).not.toBeCloseTo(gridEnd[1][0]);
        expect(gridStart[1][1]).toBeCloseTo(gridEnd[1][1]);

        fireEvent.click(screen.getByRole('button', { name: `Flip ${depthAxis.toUpperCase()} axis` }));
        expect(markerPositions()).toEqual(originalPositions);
        fireEvent.click(screen.getByRole('button', { name: `Flip ${verticalAxis.toUpperCase()} axis` }));
        const flippedPositions = markerPositions();
        expect(flippedPositions[0]).toEqual(originalPositions[0]);
        expect(flippedPositions[1][1] - flippedPositions[0][1])
            .toBeCloseTo(-(originalPositions[1][1] - originalPositions[0][1]));
    });
});

describe('recorded telemetry overview playback', () => {
    const originalObserver = global.ResizeObserver;
    const telemetry = [
        { extra: 'not drawable' },
        { ...rows[0], Physics_speed_kmh: 120, flag: false, missing: null, wheels: [21, 22], custom: { status: 'start' } },
        { extra: 'also not drawable' },
        { ...rows[1], Physics_speed_kmh: 220, flag: true, missing: null, wheels: [31, 32], custom: { status: 'finish' } },
    ];

    beforeEach(() => {
        jest.useFakeTimers();
        global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as any;
        jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
        jest.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => (
            window.setTimeout(() => callback(performance.now()), 16)
        ));
        jest.spyOn(window, 'cancelAnimationFrame').mockImplementation((id) => window.clearTimeout(id));
    });

    afterEach(() => {
        jest.restoreAllMocks();
        jest.useRealTimers();
        global.ResizeObserver = originalObserver;
    });

    it('shows every field from the original row and follows seek, play, pause, end and restart', () => {
        render(<Harness telemetry={telemetry} />);
        const overview = within(screen.getByRole('region', { name: 'Telemetry Overview' }));
        expect(overview.getAllByRole('term')).toHaveLength(Object.keys(telemetry[3]).length);
        expect(overview.getByText(/Sample 4 of 4/)).toBeInTheDocument();
        expect(overview.getByText('220')).toBeInTheDocument();
        expect(overview.getByText('Yes')).toBeInTheDocument();
        expect(overview.getByText('null')).toBeInTheDocument();
        expect(overview.getByText('[31,32]')).toBeInTheDocument();
        expect(overview.getByText('{"status":"finish"}')).toBeInTheDocument();

        const slider = within(screen.getByRole('group', { name: 'Playback position' })).getByRole('slider');
        fireEvent.keyDown(slider, { key: 'Home' });
        expect(overview.getByText(/Sample 2 of 4/)).toBeInTheDocument();
        expect(overview.getByText('120')).toBeInTheDocument();
        expect(overview.getByText('No')).toBeInTheDocument();
        fireEvent.keyDown(slider, { key: 'End' });
        expect(overview.getByText('220')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Play trajectory' }));
        expect(overview.getByText('120')).toBeInTheDocument();
        act(() => { jest.advanceTimersByTime(200); });
        fireEvent.click(screen.getByRole('button', { name: 'Pause playback' }));
        act(() => { jest.advanceTimersByTime(1500); });
        expect(overview.getByText('120')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Play trajectory' }));
        act(() => { jest.advanceTimersByTime(1200); });
        expect(overview.getByText('220')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Play trajectory' })).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Restart playback' }));
        expect(overview.getByText('120')).toBeInTheDocument();
    });

    it('keeps the feature search applied as the trajectory moves', () => {
        render(<Harness telemetry={telemetry} />);
        const overview = within(screen.getByRole('region', { name: 'Telemetry Overview' }));
        fireEvent.change(overview.getByRole('textbox', { name: 'Search telemetry features' }), { target: { value: 'SPEED' } });
        expect(overview.getAllByRole('term')).toHaveLength(1);
        expect(overview.getByText('220')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Restart playback' }));
        expect(overview.getByText('120')).toBeInTheDocument();
        expect(overview.queryByText('wheels')).not.toBeInTheDocument();
        fireEvent.change(overview.getByRole('textbox'), { target: { value: 'unknown field' } });
        expect(overview.getByText('No matching telemetry features found')).toBeInTheDocument();
    });

    it('clears stale values when the session changes or has no drawable samples', () => {
        const view = render(<Harness telemetry={telemetry} />);
        const overview = within(screen.getByRole('region', { name: 'Telemetry Overview' }));
        expect(overview.getByText('220')).toBeInTheDocument();
        view.rerender(<Harness local={false} />);
        expect(overview.queryByRole('term')).not.toBeInTheDocument();
        view.rerender(<Harness telemetry={[{ ...rows[0], Physics_speed_kmh: 77 }]} />);
        expect(overview.getByText('77')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Play trajectory' })).toBeDisabled();
        view.rerender(<Harness telemetry={[{ extra: 'no coordinates' }]} />);
        expect(overview.queryByRole('term')).not.toBeInTheDocument();
        expect(overview.getByText('No telemetry data available at this playback position')).toBeInTheDocument();
        view.rerender(<Harness selected={false} />);
        expect(overview.queryByRole('term')).not.toBeInTheDocument();
    });
});
