import { useContext } from 'react';
import { render, screen } from '@testing-library/react';
import { Theme } from '@radix-ui/themes';
import apiService from 'services/api.service';
import { AnalysisContext } from '../../analysis-context';
import MapVisualization from './MapVisualization';

jest.mock('radix-ui/internal', () => jest.requireActual('radix-ui/dist/internal.js'), { virtual: true });
jest.mock('services/api.service', () => ({ __esModule: true, default: { post: jest.fn() } }));
const mockGetCircuitMap = jest.fn().mockResolvedValue(null);
const mockGetLabelName = jest.fn();
jest.mock('contexts/CircuitMapsContext', () => ({ useCircuitMaps: () => ({ getCircuitMapByTrack: mockGetCircuitMap }) }));
jest.mock('contexts/AiLabelsContext', () => ({ useAiLabels: () => ({ getLabelName: mockGetLabelName }) }));

const playbackSummary = jest.fn();
const rows = [0, 1].map((index) => ({
    Graphics_current_time: index * 1000,
    Graphics_player_car_id: 0,
    Graphics_car_id: [0],
    Graphics_car_coordinates: [{ x: 100 + index, y: 1, z: 50 + index }],
    Physics_speed_kmh: 180 + index,
}));

const Harness = () => {
    const defaults = useContext(AnalysisContext);
    return <Theme><AnalysisContext.Provider value={{
        ...defaults,
        setRecordedPlaybackSummary: playbackSummary,
        sessionSelected: {
            storage: 'local', SessionId: 'local-ibt:1', session_name: 'spa.ibt',
            map: 'spa', car: 'GT3', user_id: '', points: [], data: rows,
        },
    }}><MapVisualization id="local-map" name="local-map" /></AnalysisContext.Provider></Theme>;
};

it('plays local telemetry without requesting a backend session or an ACC circuit map', () => {
    const originalObserver = global.ResizeObserver;
    global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as any;
    const canvas = jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    try {
        render(<Harness />);
        expect(screen.getByText('2 samples')).toBeInTheDocument();
        expect(playbackSummary).toHaveBeenLastCalledWith(expect.objectContaining({ sessionId: 'local-ibt:1', sampleCount: 2, durationSeconds: 1 }));
        expect(apiService.post).not.toHaveBeenCalled();
        expect(mockGetCircuitMap).not.toHaveBeenCalled();
        expect(screen.getByRole('button', { name: 'Run AI Analysis' })).toBeDisabled();
    } finally {
        canvas.mockRestore();
        global.ResizeObserver = originalObserver;
    }
});
