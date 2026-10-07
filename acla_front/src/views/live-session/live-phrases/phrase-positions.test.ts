import { getPhrasePositions } from './phrase-positions';
import { vision } from './test-fixtures';

describe('Live Phrases corner interpretation', () => {
    it.each(['left', 'right'] as const)('interprets a %s bend from geometry and boundary measurements', (corner) => {
        const frame = vision(0, { corner });
        expect(frame.analysis).not.toHaveProperty('cornerDirection');
        expect(getPhrasePositions(frame.analysis!, frame.geometry)).toEqual({ playerPosition: 'inside', opponentPosition: 'outside' });
        const oppositeGeometry = { ...frame.geometry!, curvaturePerM: -frame.geometry!.curvaturePerM,
            left: { ...frame.geometry!.left, coefficients: [frame.geometry!.left.coefficients[0], 0, 0] as [number, number, number] },
            right: { ...frame.geometry!.right, coefficients: [frame.geometry!.right.coefficients[0], 0, 0] as [number, number, number] } };
        expect(getPhrasePositions(frame.analysis!, oppositeGeometry)).toEqual({ playerPosition: 'outside', opponentPosition: 'inside' });
    });

    it('selects the nearest opponent ahead without mutating the published list', () => {
        const frame = vision(0);
        const far = frame.analysis!.opponents![0];
        const near = { lateralOffsetM: 2.5, longitudinalOffsetM: 10 };
        const behind = { ...far, longitudinalOffsetM: -2 };
        frame.analysis!.opponents = [far, behind, near];
        expect(getPhrasePositions(frame.analysis!, frame.geometry)).toEqual({ playerPosition: 'inside', opponentPosition: 'middle' });
        expect(frame.analysis!.opponents).toEqual([far, behind, near]);
    });

    it('withholds corner positions for straight, missing or conflicting geometry', () => {
        const frame = vision(0);
        expect(getPhrasePositions(frame.analysis!, null)).toEqual({});
        expect(getPhrasePositions(frame.analysis!, vision(0, { corner: 'straight' }).geometry)).toEqual({});
        expect(getPhrasePositions(frame.analysis!, { ...frame.geometry!, right: {
            ...frame.geometry!.right, coefficients: [0, 0, 0.003],
        } })).toEqual({});
    });

    it.each([5, 60])('leaves the opponent corner position unknown outside the fitted road at %s m', (distance) => {
        const frame = vision(0);
        frame.analysis!.opponents![0].longitudinalOffsetM = distance;
        expect(getPhrasePositions(frame.analysis!, frame.geometry)).toEqual({ playerPosition: 'inside', opponentPosition: undefined });
    });
});
