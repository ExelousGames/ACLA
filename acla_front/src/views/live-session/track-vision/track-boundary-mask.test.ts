import { createSegmentationLayers } from './segmentation-layers';
import { createTrackBoundaryMask } from './track-boundary-mask';

function outline(rows: string[], confidence = 0.9) {
    const labels = ['track', 'curb', 'grass', 'sand', 'Outfield asphalt road', 'car', 'fence', 'other', 'car interior'];
    const symbols = 'tcgsaCfoi';
    const pixels = rows.join(''), width = rows[0].length, height = rows.length;
    const layers = createSegmentationLayers({ task: 'segment', width, height, classNames: labels,
        instances: labels.map((_, classId) => ({ classId, confidence: classId === 0 ? 0.9 : confidence,
            box: [0, 0, 1, 1] as [number, number, number, number],
            mask: Uint8Array.from(pixels, (pixel) => Number(pixel === symbols[classId])) })),
    }, 0.65)!;
    const original = layers.trackMask.slice();
    const mask = createTrackBoundaryMask(layers, width, height);
    expect(layers.trackMask).toEqual(original);
    return rows.map((_, row) => Array.from(mask.slice(row * width, (row + 1) * width), (value) => value ? 't' : '.').join(''));
}

it.each(['c', 'g', 's', 'a'])('constructs edges from nearby %s labels across unlabeled gaps', (surface) => {
    expect(outline([`..${surface}.tttttt.${surface}..`])).toEqual(['...tttttttt...']);
});

it('retains track edges when nearby surface labels are absent or low confidence', () => {
    expect(outline(['....tttttt....'])).toEqual(['....tttttt....']);
    expect(outline(['..c.tttttt.g..'], 0.64)).toEqual(['....tttttt....']);
});

it.each(['C', 'f', 'o'])('does not bridge %s to reach a roadside label', (obstacle) => {
    expect(outline([`.c${obstacle}.tttttt.${obstacle}g.`])).toEqual(['....tttttt....']);
});

it('does not expand the road to distant labels', () => {
    expect(outline(['c.....tttttt.....g'])).toEqual(['......tttttt......']);
});

it('recovers a short missing section bounded by roadside labels and observed road above and below', () => {
    expect(outline(['..c.tttttt.g..', '..c........g..', '..c........g..', '..c.tttttt.g..']))
        .toEqual(Array(4).fill('...tttttttt...'));
});

it('requires two roadside edges and track observations on both ends of a missing section', () => {
    expect(outline(['..c........g..', '..c.tttttt.g..', '..c...........', '..c.tttttt.g..', '..c........g..']))
        .toEqual(['..............', '...tttttttt...', '..............', '...tttttttt...', '..............']);
    expect(outline(['..c........g..'])).toEqual(['..............']);
});

it('does not fill a long missing section or an excluded surface inside the corridor', () => {
    expect(outline(['..c.tttttt.g..', '..c........g..', '..c........g..', '..c........g..', '..c.tttttt.g..']))
        .toEqual(['...tttttttt...', '..............', '..............', '..............', '...tttttttt...']);
    expect(outline(['..c.tttttt.g..', '..c....o...g..', '..c.tttttt.g..']))
        .toEqual(['...tttttttt...', '..............', '...tttttttt...']);
});

it.each(['tttttttttttttttt....', '....tttttttttttttttt', 'tttttttttttttttttttt'])
('rejects an abrupt widening to %s without accepting later bonnet rows as a new seed', (bonnet) => {
    const road = '......tttttttt......';
    expect(outline([road, road, ...Array(6).fill(bonnet), road]))
        .toEqual([road, road, ...Array(6).fill('.'.repeat(20)), road]);
});

it('allows gradual perspective widening and bends without widening the corridor', () => {
    const rows = ['......tttttttt......', '.....tttttttttt.....', '....tttttttttttt....',
        '.....tttttttttttt...', '......tttttttttttt..'];
    expect(outline(rows)).toEqual(rows);
});

it('reacquires track after a long observation gap', () => {
    const road = '......tttttttt......', wider = '..tttttttttttttttt..', empty = '.'.repeat(20);
    expect(outline([road, empty, empty, empty, wider])).toEqual([road, empty, empty, empty, wider]);
});

it.each([1, 2])('keeps widening resistance across %s missing rows', (gap) => {
    const road = '......tttttttt......', empty = '.'.repeat(20);
    expect(outline([road, road, ...Array(gap).fill(empty), 'tttttttttttttttttttt']))
        .toEqual([road, road, ...Array(gap + 1).fill(empty)]);
});

it('cuts overlapping car interior out after construction without mistaking its holes for road narrowing', () => {
    const width = 20, height = 6, size = width * height;
    const track = { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number],
        mask: Uint8Array.from({ length: size }, (_, i) => Number(i % width >= 2 && i % width < 18)) };
    const cockpit = { ...track, classId: 1, mask: Uint8Array.from({ length: size }, (_, i) =>
        Number(Math.floor(i / width) === 2 && i % width >= 5 && i % width < 15)) };
    const original = track.mask.slice();
    for (const instances of [[track, cockpit], [cockpit, track]]) {
        const layers = createSegmentationLayers({ task: 'segment', width, height,
            classNames: ['track', ' CAR \t INTERIOR '], instances }, 0.65)!;
        const result = createTrackBoundaryMask(layers, width, height);
        expect(result).toEqual(track.mask.map((active, i) => cockpit.mask[i] ? 0 : active));
        expect(track.mask).toEqual(original);
    }
});
