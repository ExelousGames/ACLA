import type { SegmentResult, VisionResult } from './track-vision-types';
import { analyzeTrackPositions } from './track-position-analysis';
import type { CornerPosition } from './track-vision-types';
import { MODEL_LABELS, vision } from './test-fixtures';

const positions: CornerPosition[] = ['inside', 'middle', 'outside'];
describe('corner position vision geometry', () => {
    it.each([MODEL_LABELS, [...MODEL_LABELS].reverse()])('uses the uploaded label names regardless of class ordering: %j', (...classNames) => {
        const frame = vision(0, { classNames });
        frame.detections.segment!.classNames = classNames.map((label) => `\u00a0${label.toUpperCase().replace(/ /g, ' \t ')}  `);
        expect(analyzeTrackPositions(frame)).toEqual({
            cornerDirection: 'left', playerPosition: 'inside', carAhead: 1, opponentPosition: 'outside',
        });
    });

    it.each((['left', 'right'] as const).flatMap((corner) => [[1600, 900], [900, 1600], [3440, 1440]].map(([width, height]) => (
        { corner, width, height }
    ))))('uses track-mask edges in a $corner corner at $width × $height', ({ corner, width, height }) => {
        const frame = vision(0, { corner, width, height, player: 'middle', classNames: MODEL_LABELS });
        expect(analyzeTrackPositions(frame)).toEqual({
            cornerDirection: corner, playerPosition: 'middle', carAhead: 1, opponentPosition: 'outside',
        });
    });

    it.each(['clipped edges', 'low confidence', 'invalid mask'])('withholds positions when the track mask has %s', (scenario) => {
        const frame = vision(0, { classNames: MODEL_LABELS });
        const segment = frame.detections.segment as SegmentResult & VisionResult;
        const road = segment.instances[0];
        if (scenario === 'clipped edges') road.mask.fill(1);
        if (scenario === 'low confidence') road.confidence = 0.64;
        if (scenario === 'invalid mask') road.mask = new Uint8Array();
        expect(analyzeTrackPositions(frame)).toEqual({});
    });

    it.each(['curb', 'grass', 'other', 'fence', 'sand', 'Outfield asphalt road'])('excludes %s from usable track even when masks overlap', (label) => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment as SegmentResult & VisionResult;
        segment.instances.push({ ...segment.instances[1], classId: MODEL_LABELS.indexOf(label) });
        expect(analyzeTrackPositions(frame).opponentPosition).toBeUndefined();
    });

    it('does not use outfield asphalt as the racing surface', () => {
        const frame = vision(0, { classNames: MODEL_LABELS });
        const segment = frame.detections.segment as SegmentResult & VisionResult;
        segment.instances[0].classId = MODEL_LABELS.indexOf('Outfield asphalt road');
        expect(analyzeTrackPositions(frame)).toEqual({});
    });

    it('recognizes a car pack on the track without assigning an individual opponent position', () => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment as SegmentResult & VisionResult;
        const pack = segment.instances[1];
        pack.classId = MODEL_LABELS.indexOf('car pack');
        segment.instances[0].mask.forEach((_, index) => { if (pack.mask[index]) segment.instances[0].mask[index] = 0; });
        expect(analyzeTrackPositions(frame)).toEqual({ cornerDirection: 'left', playerPosition: 'inside', carAhead: 1 });
    });

    it('keeps an individually detected opponent when a car-pack mask overlaps it at the same depth', () => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment as SegmentResult & VisionResult;
        segment.instances.unshift({ ...segment.instances[1], classId: MODEL_LABELS.indexOf('car pack') });
        expect(analyzeTrackPositions(frame)).toEqual({
            cornerDirection: 'left', playerPosition: 'inside', carAhead: 1, opponentPosition: 'middle',
        });
    });

    it('does not describe a farther car as the closest opponent when a pack is ahead of it', () => {
        const frame = vision(0, { classNames: MODEL_LABELS, cars: [[0.45, 0.4, 0.52, 0.5], [0.7, 0.48, 0.77, 0.6]] });
        const segment = frame.detections.segment as SegmentResult & VisionResult;
        segment.instances[2].classId = MODEL_LABELS.indexOf('car pack');
        expect(analyzeTrackPositions(frame)).toEqual({ cornerDirection: 'left', playerPosition: 'inside', carAhead: 1 });
    });

    it.each([
        [0.25, 'inside'], [0.5, 'middle'], [0.75, 'outside'],
    ] as const)('keeps the car reference fixed as the road moves to put the car at %s track width', (across, playerPosition) => {
        const playerCenterX = 0.62;
        const detection = vision(0, { playerCenterX, cars: [],
            road: (x, y) => Math.abs(x - (playerCenterX + (0.5 - across) * 0.45 - (0.8 - y) ** 2))
                <= (0.25 + 0.5 * (y - 0.4)) / 2 });
        expect(analyzeTrackPositions(detection)).toEqual({ carAhead: 0, cornerDirection: 'left', playerPosition });
        expect(detection.playerCenterX).toBe(playerCenterX);
    });

    it.each([
        ['left', 'outside'], ['right', 'inside'],
    ] as const)('uses the car center to the right of a left-seat camera in a %s corner', (corner, playerPosition) => {
        const detection = vision(0, { corner, player: 'middle', playerCenterX: 0.62 });
        const result = analyzeTrackPositions(detection);
        expect(result).toEqual({ carAhead: 1, cornerDirection: corner, playerPosition, opponentPosition: 'outside' });
        const centeredReference = analyzeTrackPositions({ ...detection, playerCenterX: 0.5 });
        expect(centeredReference.playerPosition).toBe('middle');
        expect(centeredReference.opponentPosition).toBe(result.opponentPosition);
    });

    it.each([null, NaN, Infinity, -0.1, 0, 1, 1.2])('withholds positions without a valid car-center alignment: %s', (playerCenterX) => {
        expect(analyzeTrackPositions(vision(0, { playerCenterX }))).toEqual({});
    });

    it('follows the road under the calibrated car center when image center is off the road', () => {
        const detection = vision(0, { playerCenterX: 0.73, cars: [],
            road: (x, y) => Math.abs(x - (0.73 - (0.8 - y) ** 2)) <= 0.09 + 0.225 * (y - 0.4) });
        expect(analyzeTrackPositions(detection)).toEqual({ carAhead: 0, cornerDirection: 'left', playerPosition: 'middle' });
        expect(analyzeTrackPositions({ ...detection, playerCenterX: 0.5 })).toEqual({});
    });

    it.each((['left', 'right'] as const).flatMap((corner) => positions.flatMap((player) => positions.map((opponent) => (
        { corner, player, opponent }
    )))))('identifies both positions in a $corner corner: player $player, opponent $opponent', ({ corner, player, opponent }) => {
        expect(analyzeTrackPositions(vision(0, { corner, player, opponent }))).toEqual({
            carAhead: 1, cornerDirection: corner, playerPosition: player, opponentPosition: opponent,
        });
    });

    it.each([[1600, 900], [900, 1600], [1000, 1000], [3440, 1440]])(
        'removes letterbox padding for a %s × %s driving view', (width, height) => {
            expect(analyzeTrackPositions(vision(0, { width, height, corner: 'right', player: 'middle', playerCenterX: 0.62 }))).toEqual({
                carAhead: 1, cornerDirection: 'right', playerPosition: 'inside', opponentPosition: 'outside',
            });
        },
    );

    it.each(positions)('does not confuse an off-center straight with a corner: player %s', (player) => {
        expect(analyzeTrackPositions(vision(0, { corner: 'straight', player })).cornerDirection).toBeUndefined();
    });

    it('does not mistake a slanted straight road for a corner', () => {
        const detection = vision(0, { road: (x, y) => Math.abs(x - (0.65 - (0.8 - y) * 0.35)) < 0.14 + (y - 0.4) * 0.4 });
        expect(analyzeTrackPositions(detection).cornerDirection).toBeUndefined();
    });

    it('does not infer a corner from a bounding box without track mask evidence', () => {
        const detection = vision(0);
        const segment = detection.detections.segment as SegmentResult & VisionResult;
        segment.instances[0].mask.fill(0);
        expect(analyzeTrackPositions(detection)).toEqual({});
    });

    it('bridges an opponent mask occluding the road without changing the edges', () => {
        const detection = vision(0, { opponent: 'middle' });
        const segment = detection.detections.segment as SegmentResult & VisionResult;
        const road = segment.instances[0].mask;
        const car = segment.instances[1].mask;
        road.forEach((_, index) => { if (car[index]) road[index] = 0; });
        expect(analyzeTrackPositions(detection)).toEqual({
            carAhead: 1, cornerDirection: 'left', playerPosition: 'inside', opponentPosition: 'middle',
        });
    });

    it.each([
        ['clipped edges', (x: number, y: number) => x < 0.8 - (0.8 - y) ** 2],
        ['disconnected track', (x: number, y: number) => y < 0.65 ? x < 0.25 : x > 0.3 && x < 0.9],
        ['track absent under player', (x: number) => x < 0.3],
        ['conflicting bends', (x: number, y: number) => Math.abs(x - (0.5 + 0.1 * Math.sin((0.8 - y) * Math.PI / 0.2))) < 0.22],
    ] as const)('withholds position claims with %s', (_label, road) => {
        expect(analyzeTrackPositions(vision(0, { road }))).toEqual({});
    });

    it('does not bridge non-track objects even when their masks overlap track', () => {
        const detection = vision(0, { opponent: 'middle' });
        const segment = detection.detections.segment as SegmentResult & VisionResult;
        segment.classNames.push('kerb');
        segment.instances.push({ ...segment.instances[1], classId: 2 });
        expect(analyzeTrackPositions(detection).opponentPosition).toBeUndefined();
    });

    it.each([
        [0.05, 0.05, 0.15, 0.15], // HUD / mirror.
        [0.01, 0.45, 0.11, 0.65], // Outside the forward region.
        [0.2, 0.65, 0.8, 1], // Player's bodywork.
        [0.48, 0.25, 0.5, 0.3], // Distant, tiny car.
        [0.1, 0.45, 0.2, 0.6], // Not on the player's road.
    ])('ignores unsupported opponent detections: %s, %s, %s, %s', (...box) => {
        const result = analyzeTrackPositions(vision(0, { cars: [box as [number, number, number, number]] }));
        expect(result.carAhead).toBe(0);
        expect(result.opponentPosition).toBeUndefined();
    });

    it('selects the closest visible opponent on the same road', () => {
        const detection = vision(0, { cars: [[0.45, 0.4, 0.52, 0.5], [0.7, 0.48, 0.77, 0.6]] });
        expect(analyzeTrackPositions(detection).opponentPosition).toBe('outside');
    });

    it('accepts supported labels without case or surrounding whitespace sensitivity', () => {
        const detection = vision(0);
        detection.detections.segment!.classNames = [' ROAD ', ' Race Car '];
        expect(analyzeTrackPositions(detection)).toEqual({
            carAhead: 1, cornerDirection: 'left', playerPosition: 'inside', opponentPosition: 'outside',
        });
    });

    it.each(['track only', 'unsupported car label', 'low car confidence', 'low track confidence', 'missing mask'])('withholds opponent positions for %s', (scenario) => {
        const detection = vision(0);
        const segment = detection.detections.segment as SegmentResult & VisionResult;
        if (scenario === 'track only') segment.classNames = ['track'];
        if (scenario === 'unsupported car label') segment.classNames[1] = 'carpet';
        if (scenario === 'low car confidence') segment.instances[1].confidence = 0.64;
        if (scenario === 'low track confidence') segment.instances[0].confidence = 0.64;
        if (scenario === 'missing mask') segment.instances[0].mask = new Uint8Array();
        expect(analyzeTrackPositions(detection).opponentPosition).toBeUndefined();
    });
});
