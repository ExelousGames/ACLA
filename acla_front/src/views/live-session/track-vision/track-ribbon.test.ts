import { fitTrackRibbon, TrackRibbonPair } from './track-ribbon';

function observations(count: number, edge: (y: number, index: number) => [number, number]): TrackRibbonPair[] {
    return Array.from({ length: count }, (_, index) => {
        const y = 100 + index / (count - 1) * 600;
        const [left, right] = edge(y, index);
        return { left: { x: left, y }, right: { x: right, y } };
    });
}

it.each([2, 12, 80, 600])('fits exactly 50 paired stations from %s observations while preserving a perspective taper', (count) => {
    const source = observations(count, (y) => [400 - y * 0.4, 400 + y * 0.6]);
    const original = JSON.stringify(source);
    const ribbon = fitTrackRibbon(source)!;
    expect(ribbon.pairs).toHaveLength(50);
    expect(ribbon.pairs[0].left.y).toBe(100);
    expect(ribbon.pairs[49].left.y).toBe(700);
    expect(ribbon.pairs.slice(1).every(({ left }, index) => left.y > ribbon.pairs[index].left.y)).toBe(true);
    ribbon.pairs.forEach(({ left, right }) => {
        expect(left.y).toBe(right.y);
        expect(left.x).toBeCloseTo(400 - left.y * 0.4);
        expect(right.x).toBeCloseTo(400 + right.y * 0.6);
    });
    expect(JSON.stringify(source)).toBe(original);
});

it('reduces jagged model edges while retaining an S-bend and changing track width', () => {
    const center = (y: number) => 400 + 80 * Math.sin((y - 100) / 600 * 2 * Math.PI);
    const halfWidth = (y: number) => 20 + y * 0.15;
    const source = observations(400, (y, index) => {
        const noise = index % 2 ? 8 : -8;
        return [center(y) - halfWidth(y) + noise, center(y) + halfWidth(y) - noise];
    });
    const fitted = fitTrackRibbon(source)!.pairs;
    const errors = fitted.flatMap(({ left, right }) => [
        left.x - (center(left.y) - halfWidth(left.y)), right.x - (center(right.y) + halfWidth(right.y)),
    ]);
    expect(Math.sqrt(errors.reduce((sum, error) => sum + error ** 2, 0) / errors.length)).toBeLessThan(2);
    expect(fitted.every(({ left, right }) => Number.isFinite(left.x) && Number.isFinite(right.x) && left.x < right.x)).toBe(true);
    expect((fitted[12].left.x + fitted[12].right.x) / 2).toBeGreaterThan(470);
    expect((fitted[37].left.x + fitted[37].right.x) / 2).toBeLessThan(330);
});

it('does not construct a ribbon from an isolated row or no track evidence', () => {
    expect(fitTrackRibbon([])).toBeNull();
    expect(fitTrackRibbon([{ left: { x: 10, y: 20 }, right: { x: 30, y: 20 } }])).toBeNull();
});
