import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import { DEFAULT_CAMERA } from '../track-vision/camera-projection';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { PhraseCornerGeometry } from './phrase-corner-geometry';
import { evaluateRoad } from '../track-vision/road-polynomial';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { Closure } from './closure';
import { createPhraseRoot, PHRASE_DEFINITIONS, PhraseEngine, type PhraseContext } from './phrase-engine';

/** Control/geometry fixtures use explicit 500 ms preparation; timing tests exercise the one-second default. */
export function conditionTestEngine(...[phrases = PHRASE_DEFINITIONS, rootBuilder = createPhraseRoot]: ConstructorParameters<typeof PhraseEngine>) {
    return new PhraseEngine(phrases.map((phrase) => ({ ...phrase, preparationTimeMs: 500 })), (definitions) => {
        const root = rootBuilder(definitions);
        const configure = (node: Closure<PhraseContext>) => {
            node.metadata.preparationTimeMs = 500;
            node.children.forEach((child) => { if (child instanceof Closure) configure(child); });
        };
        configure(root);
        return root;
    });
}

type CornerPosition = 'inside' | 'middle' | 'outside';

/** Repeated variable braking on a 400 m radius turn above the 0.5 degree / 10 m minimum. */
export function variableBrakingReference(): TopLapReferenceEntry {
    let energy = 3600;
    return {
        game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_steering', 'expert_optimal_speed'],
        x: Array.from({ length: 500 }, (_, i) => i / 500),
        y: Array.from({ length: 500 }, (_, i) => {
            const phase = i % 50, braking = phase >= 15 && phase < 35;
            if (phase === 0) energy = 3600;
            energy += (braking ? -1 : 1) * (10 + (i % 7) * 5);
            const angle = i * 2 * Math.PI / 500;
            return [Math.cos(angle) * 400, Math.sin(angle) * 400, 0,
                braking ? 0.8 : 0, braking ? 0 : 0.8, 0.3, Math.sqrt(energy) * 3.6];
        }),
    };
}

/** A reference turn starts before the tagged corner, with both captured track edges. */
export function steeringCornerFixture(game: 'acc' | 'iracing' = 'acc') {
    let x = 0, ground = 0;
    const reference: TopLapReferenceEntry = {
        game, track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_steering', 'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_speed'],
        x: Array.from({ length: 100 }, (_, index) => index / 100),
        y: Array.from({ length: 100 }, (_, index) => {
            const row = [x, game === 'acc' ? ground : 0, game === 'iracing' ? ground : 0, 0, 0, 0, 36];
            // The geometry itself begins bending at 250 raw coordinate units.
            const angle = 12 * (Math.max(0, Math.min(15, index - 25)) + (index === 25 ? 0.5 : 0)) * Math.PI / 180;
            x += 10 * Math.cos(angle);
            ground += 10 * Math.sin(angle);
            return row;
        }),
    };
    const line = reference.y.map((row, index) => ({
        bin: index, normalized_position: reference.x[index], x: row[0], y: row[1], z: row[2], sample_count: 1, updated_at: '',
    }));
    const map: CircuitMapDto = {
        id: 'steering-map', game, circuit_name: 'Test circuit', source_track_key: 'test', resolution: 1000,
        samples: {
            middle_line: line,
            left_boundary: line.map((point) => ({ ...point, x: point.x - 5 })),
            right_boundary: line.map((point) => ({ ...point, x: point.x + 5 })),
        },
        centerline_segments: [{ id: 'turn-1', name: 'Turn 1', tags: ['corner'], start_position: 0.27, end_position: 0.45 }],
    };
    return { map, reference };
}

export const circuitMap = (speed: 'slow' | 'fast' = 'slow', linked = false): CircuitMapDto => ({
    id: 'test-map', game: 'acc', circuit_name: 'test', source_track_key: 'test', resolution: 1000,
    samples: { middle_line: [
        [0, 0, 0], [0.1, 1000, 0], [0.15, 1100, 100], [0.2, 1100, 200],
        [0.21, 1100, 250], [0.26, 1200, 350], [0.31, 1300, 350], [0.9, -1000, 0],
    ].map(([position, x, y], index) => ({ bin: index, normalized_position: position, x, y, z: 0, sample_count: 1, updated_at: '' })) },
    centerline_tags: [
        { id: 'turn-1', label: 'corner', start_position: 0.1, end_position: 0.2 },
        { id: 'speed-1', label: speed, start_position: 0.1, end_position: 0.2 },
        { id: 'straight', label: 'long straight', start_position: 0.4, end_position: 0.8 },
        ...(linked ? [{ id: 'turn-2', label: 'fast corner', start_position: 0.21, end_position: 0.31 }] : []),
    ],
});

/** Equal-length steps with a known heading profile, independent of the classifier. */
export const shapedCornerMap = (shape: PhraseCornerGeometry['shape']): CircuitMapDto => {
    let x = 0, y = 0;
    const samples = [{ x, y }];
    for (let index = 0; index <= 80; index++) {
        const progress = index / 80;
        const angle = shape === 'hairpin' ? Math.PI * progress
            : shape === 's-bend' ? Math.PI / 2 * Math.sin(Math.PI * progress)
                : shape === 'tightening' ? Math.PI / 2 * progress ** 2
                    : shape === 'opening' ? Math.PI / 2 * (2 * progress - progress ** 2)
                        : Math.PI / 2 * progress;
        x += 5 * Math.cos(angle);
        y += 5 * Math.sin(angle);
        samples.push({ x, y });
    }
    return {
        ...circuitMap(),
        samples: { middle_line: samples.map((point, index) => ({
            ...point, z: 0, normalized_position: 0.1 + 0.2 * index / 81,
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
