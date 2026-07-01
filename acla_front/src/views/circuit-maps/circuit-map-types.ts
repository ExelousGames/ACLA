export type CircuitMapGame = 'acc' | 'other';

export type CircuitMapCaptureMode = 'left_boundary' | 'right_boundary' | 'pit_lane';

export type FutureCircuitMapCaptureMode =
    | 'racing_line'
    | 'sector_marker'
    | 'start_finish';

export const CIRCUIT_MAP_GAMES: { value: CircuitMapGame; label: string }[] = [
    { value: 'acc', label: 'ACC' },
    { value: 'other', label: 'Other' }
];

export const CIRCUIT_MAP_CAPTURE_MODES: { value: CircuitMapCaptureMode; label: string }[] = [
    { value: 'left_boundary', label: 'Left Boundary' },
    { value: 'right_boundary', label: 'Right Boundary' },
    { value: 'pit_lane', label: 'Pit Lane' }
];

export const FUTURE_CIRCUIT_MAP_CAPTURE_MODES: FutureCircuitMapCaptureMode[] = [
    'racing_line',
    'sector_marker',
    'start_finish'
];

export type CircuitMapBinSample = {
    bin: number;
    normalized_position: number;
    x: number;
    y: number;
    z: number;
    sample_count: number;
    updated_at: string;
    locked?: boolean;
};

export type CircuitMapSamplesByMode = Partial<Record<CircuitMapCaptureMode, CircuitMapBinSample[]>>;

export type CircuitMapSummaryDto = {
    id: string;
    game: CircuitMapGame;
    circuit_name: string;
    source_track_key?: string | null;
    updated_at?: string | null;
    sample_count?: number;
};

export type CircuitMapDto = CircuitMapSummaryDto & {
    resolution: number;
    samples: CircuitMapSamplesByMode;
};

export type CircuitMapAlignedRow = {
    bin: number;
    normalized_position: number;
    left_boundary?: CircuitMapBinSample;
    right_boundary?: CircuitMapBinSample;
    pit_lane?: CircuitMapBinSample;
};
