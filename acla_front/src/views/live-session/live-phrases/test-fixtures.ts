import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { DEFAULT_CAMERA } from '../track-vision/camera-projection';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { PhraseCornerGeometry } from './phrase-corner-geometry';
import { evaluateRoad } from '../track-vision/road-polynomial';

type CornerPosition = 'inside' | 'middle' | 'outside';

export const circuitMap = (speed: 'slow' | 'fast' = 'slow', linked = false): CircuitMapDto => ({
    id: 'test-map', game: 'acc', circuit_name: 'test', source_track_key: 'test', resolution: 1000,
    samples: { middle_line: [
        [0, 0, 0], [0.1, 1000, 0], [0.15, 1100, 100], [0.2, 1100, 200],
        [0.21, 1100, 250], [0.26, 1200, 350], [0.31, 1300, 350], [0.9, -1000, 0],
    ].map(([position, x, z], index) => ({ bin: index, normalized_position: position, x, y: 0, z, sample_count: 1, updated_at: '' })) },
    centerline_tags: [
        { id: 'turn-1', label: 'corner', start_position: 0.1, end_position: 0.2 },
        { id: 'speed-1', label: speed, start_position: 0.1, end_position: 0.2 },
        { id: 'straight', label: 'long straight', start_position: 0.4, end_position: 0.8 },
        ...(linked ? [{ id: 'turn-2', label: 'fast corner', start_position: 0.21, end_position: 0.31 }] : []),
    ],
});

/** Equal-length steps with a known heading profile, independent of the classifier. */
export const shapedCornerMap = (shape: PhraseCornerGeometry['shape']): CircuitMapDto => {
    let x = 0, z = 0;
    const samples = [{ x, z }];
    for (let index = 0; index <= 80; index++) {
        const progress = index / 80;
        const angle = shape === 'hairpin' ? Math.PI * progress
            : shape === 's-bend' ? Math.PI / 2 * Math.sin(Math.PI * progress)
                : shape === 'tightening' ? Math.PI / 2 * progress ** 2
                    : shape === 'opening' ? Math.PI / 2 * (2 * progress - progress ** 2)
                        : Math.PI / 2 * progress;
        x += 5 * Math.cos(angle);
        z += 5 * Math.sin(angle);
        samples.push({ x, z });
    }
    return {
        ...circuitMap(),
        samples: { middle_line: samples.map((point, index) => ({
            ...point, y: 0, normalized_position: 0.1 + 0.2 * index / 81,
            bin: index, sample_count: 1, updated_at: '',
        })) },
        centerline_segments: [{ id: 'shaped-turn', tags: ['corner', 'slow'], start_position: 0.1, end_position: 0.3 }],
    };
};

/** Published measurements only: phrase tests do not need screen masks. */
export function vision(capturedAt: number, options: {
    corner?: 'left' | 'right' | 'straight';
    player?: CornerPosition;
    opponent?: CornerPosition;
    cameraOffset?: number | null;
    carAhead?: boolean;
} = {}): TrackVisionDetection {
    const { corner = 'left', player = 'inside', opponent = 'outside', carAhead = true } = options;
    const distances = (position: CornerPosition) => {
        const left = { inside: 2.5, middle: 5, outside: 7.5 }[position];
        return { leftBoundaryDistanceM: corner === 'right' ? 10 - left : left,
            rightBoundaryDistanceM: corner === 'right' ? left : 10 - left };
    };
    const curve = corner === 'straight' ? 0 : corner === 'left' ? -0.003 : 0.003;
    const road = { coefficients: [0, 0, curve] as [number, number, number], minY: 8, maxY: 50, rmseM: 0 };
    const driver = distances(player), other = distances(opponent);
    const edge = (x: number) => ({ ...road, coefficients: [x + 64 * curve, -16 * curve, curve] as [number, number, number] });
    const boundary = (x: number) => Array.from({ length: 43 }, (_, index) => ({
        x: evaluateRoad(edge(x), index + 8), y: index + 8, z: 0,
    }));
    return {
        capturedAt, width: 1600, height: 900, detections: {},
        birdsEyeScene: options.cameraOffset === null ? null : {
            leftBoundary: [boundary(-driver.leftBoundaryDistanceM)], rightBoundary: [boundary(driver.rightBoundaryDistanceM)],
            centerline: [boundary(5 - driver.leftBoundaryDistanceM)], unplacedCars: 0,
            cars: carAhead ? [{ classId: 3, confidence: 0.9, pack: false, position: {
                x: other.leftBoundaryDistanceM - driver.leftBoundaryDistanceM + curve * 100, y: 18, z: 0,
            } }] : [],
        },
        geometry: options.cameraOffset === null ? null : {
            leftBoundary: [], rightBoundary: [], left: edge(-driver.leftBoundaryDistanceM),
            right: edge(driver.rightBoundaryDistanceM), center: edge(5 - driver.leftBoundaryDistanceM),
            referenceY: 8, trackWidthM: 10, lateralOffsetM: 0, headingDeg: 0, curvaturePerM: 2 * curve,
        }, reconstruction: null,
        calibration: options.cameraOffset === null ? undefined : { ...DEFAULT_CAMERA, imageWidth: 1600, imageHeight: 900, lateralOffsetM: options.cameraOffset ?? 0 },
    };
}
