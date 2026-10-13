import { useContext } from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Theme } from '@radix-ui/themes';
import apiService from 'services/api.service';
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

const Harness = ({ local = true, telemetry = rows, selected = true }: {
    local?: boolean;
    telemetry?: Record<string, any>[];
    selected?: boolean;
}) => {
    const defaults = useContext(AnalysisContext);
    const session = selected ? {
        storage: local ? 'local' as const : 'cloud' as const,
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
