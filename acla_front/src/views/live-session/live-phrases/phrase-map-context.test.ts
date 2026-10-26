import { createPhraseMapContext } from './phrase-map-context';
import { circuitMap } from './test-fixtures';

describe('Live Map phrase context', () => {
    it('requires a nearby long straight after the corner without an intervening corner', () => {
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'turn', tags: ['corner', 'slow'], start_position: 0.1, end_position: 0.2 },
            { id: 'after', tags: ['long straight'], start_position: 0.2, end_position: 0.4 },
        ];
        const after = map.centerline_segments[1];
        expect(createPhraseMapContext(map).approaching(0.05)?.longStraightAfterCorner).toBe(1);
        after.start_position = 0.205;
        expect(createPhraseMapContext(map).approaching(0.05)?.longStraightAfterCorner).toBe(1);
        after.tags = ['straight'];
        expect(createPhraseMapContext(map).approaching(0.05)?.longStraightAfterCorner).toBe(0);
        after.tags = ['long straight'];
        after.start_position = 0.3;
        expect(createPhraseMapContext(map).approaching(0.05)?.longStraightAfterCorner).toBe(0);
        after.start_position = 0.21;
        map.centerline_segments.push({ id: 'intervening', tags: ['corner'], start_position: 0.202, end_position: 0.208 });
        expect(createPhraseMapContext(map).approaching(0.05)?.longStraightAfterCorner).toBe(0);
    });

    it('resolves a long straight after a corner across start/finish and rejects an approach-only straight', () => {
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'turn', tags: ['slow corner'], start_position: 0.9, end_position: 0.99 },
            { id: 'after', tags: ['long straight'], start_position: 0.99, end_position: 0.2 },
        ];
        expect(createPhraseMapContext(map).approaching(0.8)?.longStraightAfterCorner).toBe(1);
        map.centerline_segments[0].end_position = 0.01;
        map.centerline_segments[1].start_position = 0.01;
        expect(createPhraseMapContext(map).approaching(0.8)?.longStraightAfterCorner).toBe(1);
        map.centerline_segments[1] = { id: 'before', tags: ['long straight'], start_position: 0.8, end_position: 0.9 };
        expect(createPhraseMapContext(map).approaching(0.85)?.longStraightAfterCorner).toBe(0);
    });
    it.each(['corner', 'slow corner', 'fast corner'])('requires actual bounds for %s membership, including overlapping and wraparound corners', (label) => {
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'turn', tags: [label], start_position: 0.1, end_position: 0.2 },
            { id: 'overlap', tags: ['corner'], start_position: 0.15, end_position: 0.25 },
            { id: 'wrap', tags: ['corner'], start_position: 0.95, end_position: 0.05 },
            { id: 'sequence', tags: ['corner', 'consecutive corners'], start_position: 0.1, end_position: 0.3 },
        ];
        const context = createPhraseMapContext(map);
        expect(context.at(0.09).phase).toBe('entry');
        expect(context.cornersAt(0.09)).toEqual([]);
        expect(context.cornersAt(0.1)).toEqual(['turn']);
        expect(context.cornersAt(0.18)).toEqual(['turn', 'overlap']);
        expect(context.cornersAt(0.2)).toEqual(['turn', 'overlap']);
        expect(context.cornersAt(0.201)).toEqual(['overlap']);
        expect(context.cornersAt(0.26)).toEqual([]);
        expect(context.cornersAt(0.97)).toEqual(['wrap']);
        expect(context.cornersAt(0)).toEqual(['wrap']);
        expect(context.cornersAt(0.05)).toEqual(['wrap']);
        expect(context.cornersAt(0.051)).toEqual([]);
        expect(context.cornersAt(NaN)).toBeUndefined();
        expect(context.cornersAt(undefined)).toBeUndefined();
        expect(createPhraseMapContext(null).cornersAt(0.15)).toBeUndefined();
    });

    it('resolves long-straight membership independently of corner approach, including overlapping and wraparound tags', () => {
        const map = circuitMap();
        map.centerline_tags!.push(
            { id: 'wrap-straight', label: 'long straight', start_position: 0.95, end_position: 0.1 },
            { id: 'overlap', label: 'long straight', start_position: 0.05, end_position: 0.09 },
            { id: 'ordinary', label: 'straight', start_position: 0.25, end_position: 0.35 },
        );
        const context = createPhraseMapContext(map);
        expect(context.longStraightsAt(0.97)).toEqual(['wrap-straight']);
        expect(context.longStraightsAt(0)).toEqual(['wrap-straight']);
        expect(context.longStraightsAt(0.08)).toEqual(['wrap-straight', 'overlap']);
        expect(context.at(0.099).phase).toBe('entry');
        expect(context.longStraightsAt(0.099)).toEqual(['wrap-straight']);
        expect(context.longStraightsAt(0.1)).toEqual(['wrap-straight']);
        expect(context.longStraightsAt(0.101)).toEqual([]);
        expect(context.longStraightsAt(0.3)).toEqual([]);
        expect(context.longStraightsAt(NaN)).toBeUndefined();
        expect(context.longStraightsAt(undefined)).toBeUndefined();
        expect(createPhraseMapContext(null).longStraightsAt(0.5)).toBeUndefined();
    });

    it('resolves all tags on a segment and respects a cleared segment list', () => {
        const map = { ...circuitMap(), centerline_segments: [
            { id: 'turn', tags: ['corner', 'fast'], start_position: 0.1, end_position: 0.2 },
            { id: 'straight', tags: ['long straight'], start_position: 0.4, end_position: 0.8 },
        ] };
        const context = createPhraseMapContext(map);
        expect(context.at(0.11)).toMatchObject({ sectionId: 'turn', phase: 'entry', cornerSpeed: 'fast' });
        expect(context.at(0.5)).toEqual({ sectionId: 'straight', phase: 'straight' });
        map.centerline_segments[0].tags.push('slow');
        expect(createPhraseMapContext(map).at(0.11).cornerSpeed).toBeUndefined();
        expect(createPhraseMapContext({ ...map, centerline_segments: [] }).ready).toBe(false);
    });

    it('uses tagged corner speed and lap progress, including the approach', () => {
        const context = createPhraseMapContext(circuitMap());
        expect(context.ready).toBe(true);
        expect(context.at(0.09)).toMatchObject({ phase: 'entry', cornerSpeed: 'slow' });
        expect(context.at(0.11)).toMatchObject({ phase: 'entry', cornerSpeed: 'slow' });
        expect(context.at(0.15)).toMatchObject({ phase: 'middle', cornerSpeed: 'slow' });
        expect(context.at(0.18)).toMatchObject({ phase: 'exit', cornerSpeed: 'slow' });
        expect(context.at(0.5)).toEqual({ sectionId: 'straight', phase: 'straight' });
        expect(context.at(0.35)).toEqual({});
    });

    it('resolves the next corner for time-based arrival beyond the fixed-distance lookahead', () => {
        const map = circuitMap('slow', true);
        map.centerline_tags!.push({ id: 'sequence', label: 'consecutive corners', start_position: 0.1, end_position: 0.31 });
        const context = createPhraseMapContext(map);
        expect(context.at(0.05)).toEqual({});
        expect(context.approaching(0.05)).toMatchObject({
            sectionId: 'turn-1', cornerStartPosition: 0.1, cornerSpeed: 'slow',
            sequenceId: 'sequence', sequenceRemaining: 1, linkedOpposite: 1,
        });
        expect(context.approaching(0.1)).toBeUndefined();
        expect(context.approaching(0.15)).toBeUndefined();
        expect(context.approaching(NaN)).toBeUndefined();
        expect(context.approaching(undefined)).toBeUndefined();
    });

    it.each([
        ['acc', 1, 'right'], ['acc', -1, 'left'],
        ['iracing', -1, 'right'], ['iracing', 1, 'left'],
    ] as const)('resolves the upcoming corner direction in %s with ground-axis scale %s as %s', (game, scale, direction) => {
        const map = circuitMap('slow', true);
        map.game = game;
        map.samples.middle_line = map.samples.middle_line!.map((point) => game === 'acc'
            ? { ...point, y: point.y * scale } : { ...point, y: 0, z: point.y * scale });
        const context = createPhraseMapContext(map);
        expect(context.approaching(0.05)).toMatchObject({
            sectionId: 'turn-1', cornerStartPosition: 0.1, cornerDirection: direction,
        });
        expect(context.approaching(0.205)?.cornerDirection).toBe(direction === 'left' ? 'right' : 'left');
    });

    it('leaves upcoming direction unknown when the mapped corner has no geometry', () => {
        const map = circuitMap();
        map.samples.middle_line!.forEach((point) => { point.y = 0; });
        expect(createPhraseMapContext(map).approaching(0.05)?.cornerDirection).toBeUndefined();
    });

    it.each(['slow corner', 'fast corner'])('supports the legacy %s tag without a separate speed tag', (label) => {
        const map = circuitMap();
        map.centerline_tags = [{ id: 'legacy', label, start_position: 0.1, end_position: 0.2 }];
        expect(createPhraseMapContext(map).at(0.11).cornerSpeed).toBe(label.split(' ')[0]);
    });

    it('does not infer corner speed from current driving speed or contradictory tags', () => {
        const map = circuitMap();
        map.centerline_tags = map.centerline_tags!.filter((tag) => tag.label !== 'slow');
        expect(createPhraseMapContext(map).at(0.11).cornerSpeed).toBeUndefined();
        map.centerline_tags.push(
            { id: 'slow', label: 'slow', start_position: 0.1, end_position: 0.2 },
            { id: 'fast', label: 'fast', start_position: 0.1, end_position: 0.2 },
        );
        expect(createPhraseMapContext(map).at(0.11).cornerSpeed).toBeUndefined();
    });

    it('handles corner ranges across start/finish and unordered tags', () => {
        const map = circuitMap();
        map.centerline_tags = [
            { id: 'speed', label: 'slow', start_position: 0.95, end_position: 0.05 },
            { id: 'wrap', label: 'corner', start_position: 0.95, end_position: 0.05 },
        ];
        const context = createPhraseMapContext(map);
        expect(context.at(0.94)).toMatchObject({ sectionId: 'wrap', phase: 'entry' });
        expect(context.at(0.97)).toMatchObject({ phase: 'entry', cornerSpeed: 'slow' });
        expect(context.at(0)).toMatchObject({ phase: 'middle', cornerSpeed: 'slow' });
        expect(context.at(0.04)).toMatchObject({ phase: 'exit', cornerSpeed: 'slow' });
    });

    it.each(['acc', 'iracing'] as const)('recognizes opposite linked turns in %s coordinates', (game) => {
        const map = circuitMap('slow', true);
        map.game = game;
        if (game === 'iracing') map.samples.middle_line = map.samples.middle_line!.map((point) => ({ ...point, y: 0, z: -point.y }));
        map.centerline_tags!.reverse();
        expect(createPhraseMapContext(map).at(0.11).linkedOpposite).toBe(1);
        expect(createPhraseMapContext(map).at(0.26).linkedOpposite).toBe(0);
    });

    it('withholds linked-turn guidance for the same direction, distant turns or missing geometry', () => {
        const same = circuitMap('slow', true);
        same.samples.middle_line!.find((point) => point.normalized_position === 0.31)!.x = 1100;
        expect(createPhraseMapContext(same).at(0.11).linkedOpposite).toBe(0);
        const distant = circuitMap('slow', true);
        distant.centerline_tags!.find((tag) => tag.id === 'turn-2')!.start_position = 0.26;
        expect(createPhraseMapContext(distant).at(0.11).linkedOpposite).toBe(0);
        const missing = circuitMap('slow', true);
        missing.samples.middle_line = [];
        expect(createPhraseMapContext(missing).ready).toBe(false);
        expect(createPhraseMapContext(missing).at(0.11)).toEqual({});
    });

    it('ignores malformed ranges and never treats untagged track as a straight', () => {
        const map = circuitMap();
        map.centerline_tags = [
            { id: 'bad', label: 'corner', start_position: NaN, end_position: 0.2 },
            { id: 'empty', label: 'straight', start_position: 0.1, end_position: 0.1 },
        ];
        expect(createPhraseMapContext(map).ready).toBe(false);
        expect(createPhraseMapContext(map).at(0.11)).toEqual({});
    });

    it('finds corner-tagged children inside a consecutive-corners area and honors the area beyond the proximity limit', () => {
        const map = circuitMap('slow', true);
        map.samples.middle_line!.forEach((point) => { if (point.normalized_position >= 0.21 && point.normalized_position <= 0.31) point.normalized_position += 0.05; });
        map.centerline_segments = [
            { id: 'sequence', tags: ['consecutive corners', 'corner'], start_position: 0.08, end_position: 0.38 },
            { id: 'second', tags: ['corner', 'fast'], start_position: 0.26, end_position: 0.36 },
            { id: 'partial', tags: ['corner'], start_position: 0.37, end_position: 0.45 },
            { id: 'speed', tags: ['slow'], start_position: 0.1, end_position: 0.2 },
            { id: 'first', tags: ['corner'], start_position: 0.1, end_position: 0.2 },
        ];
        const context = createPhraseMapContext(map);
        expect(context.at(0.11)).toMatchObject({
            sectionId: 'first', cornerSpeed: 'slow', linkedOpposite: 1,
            sequenceId: 'sequence', sequenceShape: 'alternating', sequenceCornerCount: 2, sequenceCornerIndex: 1, sequenceRemaining: 1,
        });
        expect(context.at(0.21)).toMatchObject({ sectionId: 'sequence', sequenceShape: 'alternating' });
        expect(context.at(0.21).phase).toBeUndefined();
        expect(context.at(0.255)).toMatchObject({ sectionId: 'second', phase: 'entry', sequenceCornerIndex: 2 });
        expect(context.at(0.3)).toMatchObject({ linkedOpposite: 0, sequenceRemaining: 0, cornerSpeed: 'fast' });
        expect(context.at(0.44).sequenceId).toBeUndefined();
    });

    it.each([false, true])('classifies same-direction members with lap wraparound: %s', (wrap) => {
        const map = circuitMap('slow', true);
        map.samples.middle_line!.find((point) => point.normalized_position === 0.31)!.x = 1100;
        map.centerline_tags!.push({ id: 'sequence', label: 'consecutive corners', start_position: 0.1, end_position: 0.31 });
        if (wrap) {
            map.samples.middle_line!.forEach((point) => { point.normalized_position = (point.normalized_position + 0.8) % 1; });
            map.centerline_tags!.forEach((tag) => {
                tag.start_position = (tag.start_position + 0.8) % 1;
                tag.end_position = (tag.end_position + 0.8) % 1;
            });
        }
        const context = createPhraseMapContext(map);
        expect(context.at(wrap ? 0.91 : 0.11)).toMatchObject({ linkedSameDirection: 1, linkedOpposite: 0,
            sequenceShape: 'same-direction', sequenceCornerIndex: 1, sequenceCornerCount: 2 });
        expect(context.at(wrap ? 0.06 : 0.26)).toMatchObject({ linkedSameDirection: 0, sequenceCornerIndex: 2, sequenceRemaining: 0 });
    });

    it('classifies a mixed sequence using all children, not just the first pair', () => {
        const map = circuitMap('slow', true);
        map.samples.middle_line!.push(...[
            [0.32, 1350, 350], [0.37, 1450, 250], [0.42, 1450, 150],
        ].map(([position, x, y], index) => ({ ...map.samples.middle_line![0], normalized_position: position, bin: 20 + index, x, y })));
        map.centerline_tags!.push(
            { id: 'third', label: 'corner', start_position: 0.32, end_position: 0.42 },
            { id: 'sequence', label: 'consecutive corners', start_position: 0.1, end_position: 0.42 },
        );
        expect(createPhraseMapContext(map).at(0.11)).toMatchObject({ sequenceShape: 'mixed', sequenceCornerCount: 3, sequenceRemaining: 2 });
        expect(createPhraseMapContext(map).at(0.26)).toMatchObject({ linkedSameDirection: 1, sequenceCornerIndex: 2 });
    });

    it('does not invent a sequence shape from area tags alone or unknown child geometry', () => {
        const map = circuitMap();
        map.centerline_segments = [{ id: 'area', tags: ['consecutive corners'], start_position: 0.1, end_position: 0.31 }];
        expect(createPhraseMapContext(map).at(0.15)).toMatchObject({ sequenceId: 'area', sequenceCornerCount: 0 });
        expect(createPhraseMapContext(map).at(0.15).phase).toBeUndefined();
        map.centerline_segments.push(
            { id: 'first', tags: ['corner'], start_position: 0.1, end_position: 0.2 },
            { id: 'unknown', tags: ['corner'], start_position: 0.22, end_position: 0.24 },
        );
        expect(createPhraseMapContext(map).at(0.11)).toMatchObject({ linkedOpposite: 0, linkedSameDirection: 0, sequenceCornerCount: 2 });
        expect(createPhraseMapContext(map).at(0.11).sequenceShape).toBeUndefined();
    });

    it('uses the smallest containing area and does not link its final corner outside that area', () => {
        const map = circuitMap('slow', true);
        map.centerline_tags!.push(
            { id: 'whole-lap', label: 'consecutive corners', start_position: 0, end_position: 1 },
            { id: 'inner', label: 'consecutive corners', start_position: 0.09, end_position: 0.205 },
        );
        const context = createPhraseMapContext(map);
        expect(context.at(0.11)).toMatchObject({ sequenceId: 'inner', sequenceCornerCount: 1, sequenceRemaining: 0, linkedOpposite: 0 });
        expect(context.at(0.26)).toMatchObject({ sequenceId: 'whole-lap', sequenceCornerCount: 2, sequenceRemaining: 0 });
    });
});
