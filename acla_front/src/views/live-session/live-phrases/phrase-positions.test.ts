import { getPhrasePositions } from './phrase-positions';
import { vision } from './test-fixtures';

const scene = (options: Parameters<typeof vision>[1] = {}) => vision(0, options).birdsEyeScene!;

describe('Live Phrases BEV interpretation', () => {
    it.each(['left', 'right'] as const)('interprets positions in a %s bend from BEV boundaries', (corner) => {
        for (const player of ['inside', 'middle', 'outside'] as const) {
            for (const opponent of ['inside', 'middle', 'outside'] as const) {
                expect(getPhrasePositions(scene({ corner, player, opponent }))).toEqual({ carAhead: 1, playerPosition: player, opponentPosition: opponent });
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
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 1, playerPosition: 'inside', opponentPosition: 'middle' });
        expect(JSON.stringify(ground)).toBe(original);
    });

    it('withholds corner positions for straight, missing or conflicting boundaries', () => {
        expect(getPhrasePositions(null)).toEqual({});
        expect(getPhrasePositions(scene({ corner: 'straight' }))).toEqual({ carAhead: 1 });
        const ground = scene();
        ground.rightBoundary = scene({ corner: 'right', player: 'outside' }).rightBoundary;
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 1 });
        ground.rightBoundary = [];
        expect(getPhrasePositions(ground)).toEqual({ carAhead: undefined });
    });

    it.each([5, 60])('does not place an opponent beyond visible boundaries at %s m', (distance) => {
        const ground = scene();
        ground.cars[0].position.y = distance;
        expect(getPhrasePositions(ground)).toEqual({ carAhead: undefined, playerPosition: 'inside' });
    });

    it('does not interpolate across a gap in the displayed boundaries', () => {
        const ground = scene();
        for (const side of ['leftBoundary', 'rightBoundary'] as const) {
            const points = ground[side][0];
            ground[side] = [points.filter(({ y }) => y <= 10), points.filter(({ y }) => y >= 25)];
        }
        expect(getPhrasePositions(ground).carAhead).toBeUndefined();
        expect(getPhrasePositions(ground).opponentPosition).toBeUndefined();
    });

    it('uses car packs for traffic ahead without inventing an individual position', () => {
        const ground = scene();
        ground.cars[0].pack = true;
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 1, playerPosition: 'inside' });
    });

    it('distinguishes empty track from unplaced traffic and ignores off-track cars', () => {
        const ground = scene({ carAhead: false });
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 0, playerPosition: 'inside' });
        ground.unplacedCars = 1;
        expect(getPhrasePositions(ground).carAhead).toBeUndefined();
        ground.unplacedCars = 0;
        ground.cars = scene().cars;
        ground.cars[0].position.x = -20;
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 0, playerPosition: 'inside' });
    });

    it('recognizes traffic on a short visible road even without enough curvature evidence', () => {
        const ground = scene();
        ground.leftBoundary = [ground.leftBoundary[0].filter(({ y }) => y >= 16 && y <= 20)];
        ground.rightBoundary = [ground.rightBoundary[0].filter(({ y }) => y >= 16 && y <= 20)];
        expect(getPhrasePositions(ground)).toEqual({ carAhead: 1 });
    });
});
