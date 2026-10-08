import { getPhrasePositions } from './phrase-positions';
import { vision } from './test-fixtures';

const scene = (options: Parameters<typeof vision>[1] = {}) => vision(0, options).birdsEyeScene!;

describe('Live Phrases BEV interpretation', () => {
    it.each(['left', 'right'] as const)('interprets positions in a %s bend from BEV boundaries', (corner) => {
        const edges = corner === 'left'
            ? { inside: 'left', middle: 'middle', outside: 'right' }
            : { inside: 'right', middle: 'middle', outside: 'left' };
        for (const player of ['inside', 'middle', 'outside'] as const) {
            for (const opponent of ['inside', 'middle', 'outside'] as const) {
                expect(getPhrasePositions(scene({ corner, player, opponent }))).toMatchObject({
                    carAhead: 1, playerCorner: corner, opponentCorner: corner,
                    playerPosition: edges[player], opponentPosition: edges[opponent],
                });
            }
        }
    });

    it('selects the nearest on-track individual ahead without mutating the published scene', () => {
        const ground = scene();
        const far = ground.cars[0];
        const near = { ...far, position: { x: 2.5, y: 10, z: 0 } };
        const behind = { ...far, position: { x: 0, y: -2, z: 0 } };
        const offTrack = { ...far, position: { x: -20, y: 9, z: 0 } };
        const pack = { ...far, pack: true, position: { x: 0, y: 9, z: 0 } };
        ground.cars = [far, behind, offTrack, pack, near];
        const original = JSON.stringify(ground);
        expect(getPhrasePositions(ground)).toMatchObject({ carAhead: 1, playerCorner: 'left', opponentCorner: 'left', playerPosition: 'left', opponentPosition: 'middle',
            opponentDistanceM: Math.hypot(2.5, 10), opponentLateralOffsetM: 2.5 });
        expect(JSON.stringify(ground)).toBe(original);
    });

    it('keeps track positions independent of straight, missing or conflicting curvature', () => {
        expect(getPhrasePositions(null)).toEqual({});
        expect(getPhrasePositions(scene({ corner: 'straight' }))).toEqual({
            carAhead: 1, playerPosition: 'left', opponentPosition: 'right',
            playerCorner: undefined, opponentCorner: undefined,
            opponentDistanceM: Math.hypot(5, 18), opponentLateralOffsetM: 5,
        });
        const ground = scene();
        ground.rightBoundary = scene({ corner: 'right', player: 'outside' }).rightBoundary;
        expect(getPhrasePositions(ground)).toEqual({
            carAhead: 1, playerPosition: 'left', opponentPosition: 'right',
            playerCorner: undefined, opponentCorner: undefined,
            opponentDistanceM: Math.hypot(4.7, 18), opponentLateralOffsetM: 4.7,
        });
        ground.rightBoundary = [];
        expect(getPhrasePositions(ground)).toEqual({ carAhead: undefined });
    });

    it.each([5, 60])('does not place an opponent beyond visible boundaries at %s m', (distance) => {
        const ground = scene();
        ground.cars[0].position.y = distance;
        expect(getPhrasePositions(ground)).toEqual({ carAhead: undefined, playerCorner: 'left', playerPosition: 'left' });
    });

    it('does not interpolate across a gap in the displayed boundaries', () => {
        const ground = scene();
        for (const side of ['leftBoundary', 'rightBoundary'] as const) {
            const points = ground[side][0];
            ground[side] = [points.filter(({ y }) => y <= 10), points.filter(({ y }) => y >= 25)];
        }
        expect(getPhrasePositions(ground).carAhead).toBeUndefined();
        expect(getPhrasePositions(ground).opponentPosition).toBeUndefined();
        expect(getPhrasePositions(ground).opponentCorner).toBeUndefined();
    });

    it('uses car packs for traffic ahead without inventing an individual position', () => {
        const ground = scene();
        ground.cars[0].pack = true;
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 1, playerCorner: 'left', playerPosition: 'left' });
    });

    it('distinguishes empty track from unplaced traffic and ignores off-track cars', () => {
        const ground = scene({ carAhead: false });
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 0, playerCorner: 'left', playerPosition: 'left' });
        ground.unplacedCars = 1;
        expect(getPhrasePositions(ground).carAhead).toBeUndefined();
        ground.unplacedCars = 0;
        ground.cars = scene().cars;
        ground.cars[0].position.x = -20;
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 0, playerCorner: 'left', playerPosition: 'left' });
    });

    it('recognizes traffic on a short visible road even without enough curvature evidence', () => {
        const ground = scene();
        ground.leftBoundary = [ground.leftBoundary[0].filter(({ y }) => y >= 16 && y <= 20)];
        ground.rightBoundary = [ground.rightBoundary[0].filter(({ y }) => y >= 16 && y <= 20)];
        expect(getPhrasePositions(ground)).toEqual({
            carAhead: 1, playerPosition: 'left', opponentPosition: 'right',
            playerCorner: undefined, opponentCorner: undefined,
            opponentDistanceM: Math.hypot(4.7, 18), opponentLateralOffsetM: 4.7,
        });
    });

    it('uses each car\'s local boundary section for its turn direction', () => {
        const ground = scene();
        const next = scene({ corner: 'right' });
        for (const side of ['leftBoundary', 'rightBoundary'] as const) {
            ground[side] = [ground[side][0].filter(({ y }) => y <= 20),
                next[side][0].map((point) => ({ ...point, y: point.y + 30 }))];
        }
        ground.cars = next.cars.map((car) => ({ ...car, position: { ...car.position, y: car.position.y + 30 } }));
        expect(getPhrasePositions(ground)).toMatchObject({
            carAhead: 1, playerCorner: 'left', playerPosition: 'left',
            opponentCorner: 'right', opponentPosition: 'left',
        });
    });

    it('evaluates the driver at the origin when remembered boundaries extend behind the car', () => {
        const ground = scene({ carAhead: false });
        // At y = 0 the driver is inside this left bend; the slice behind the car says middle.
        const ys = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
        ground.leftBoundary = [ys.map((y) => ({ x: -2.5 + 0.6 * y - 0.01 * y * y, y, z: 0, observedAt: 0 }))];
        ground.rightBoundary = [ground.leftBoundary[0].map((point) => ({ ...point, x: point.x + 10 }))];
        expect(getPhrasePositions(ground)).toMatchObject({ playerCorner: 'left', playerPosition: 'left' });
    });
});
