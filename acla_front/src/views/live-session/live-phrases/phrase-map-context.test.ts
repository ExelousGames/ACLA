import { createPhraseMapContext } from './phrase-map-context';
import { circuitMap } from './test-fixtures';

describe('Live Map phrase context', () => {
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
        if (game === 'iracing') map.samples.middle_line!.forEach((point) => { point.z *= -1; });
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
        ].map(([position, x, z], index) => ({ ...map.samples.middle_line![0], normalized_position: position, bin: 20 + index, x, z })));
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
