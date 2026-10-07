import { BadRequestException } from '@nestjs/common';
import { model } from 'mongoose';
import { CircuitMapSchema } from 'src/schemas/circuit-map.schema';
import { CircuitMapService } from './circuit-map.service';

describe('Circuit map centerline segments', () => {
    const mapId = '507f1f77bcf86cd799439011';
    const MapModel = model('CircuitMapTags', CircuitMapSchema);
    const tag = { id: 'turn-1', tags: ['corner', 'slow'], start_position: 0.1, end_position: 0.25 };
    const wrapTag = { id: 'finish', tags: ['corner', 'fast'], start_position: 0.95, end_position: 0.05 };
    const payload = {
        game: 'acc' as const, circuit_name: 'Tagged Circuit', centerline_segments: [tag, wrapTag],
        samples: { middle_line: [{ bin: 100, normalized_position: 0.1, x: 10, y: 0, z: 20, sample_count: 1, updated_at: '2026-10-06' }] },
    };
    let stored: any;
    let storage: { create: jest.Mock; findById: jest.Mock; findByIdAndUpdate: jest.Mock };
    let service: CircuitMapService;

    beforeEach(() => {
        stored = { _id: mapId, ...payload };
        storage = {
            create: jest.fn(async (data) => {
                const document = new MapModel({ _id: mapId, ...data });
                await document.validate();
                stored = document.toObject();
                return document;
            }),
            findById: jest.fn(() => ({ lean: () => ({ exec: async () => stored }) })),
            findByIdAndUpdate: jest.fn((_id, data) => ({ lean: () => ({ exec: async () => {
                stored = new MapModel({ ...stored, ...data }).toObject();
                return stored;
            } }) })),
        };
        service = new CircuitMapService(storage as any);
    });

    it('persists normal and wrapping ranges through the schema and reloads them with the map', async () => {
        expect((await service.create(payload)).centerline_segments).toEqual([tag, wrapTag]);
        expect((await service.get(mapId)).centerline_segments).toEqual([tag, wrapTag]);
        const updated = { ...tag, tags: ['  corner  ', 'fast', 'fast'] };
        const result = await service.update(mapId, { ...payload, centerline_segments: [updated] });
        expect(result.centerline_segments).toEqual([{ ...tag, tags: ['corner', 'fast'] }]);
        expect((await service.get(mapId)).centerline_segments).toEqual(result.centerline_segments);
        expect(result.samples.middle_line).toEqual(payload.samples.middle_line);
    });

    it('clears removed tags and preserves tags when an older client omits the field', async () => {
        const { centerline_segments, ...legacyPayload } = payload;
        expect((await service.update(mapId, legacyPayload)).centerline_segments).toEqual(centerline_segments);
        expect(storage.findByIdAndUpdate.mock.calls[0][1]).not.toHaveProperty('centerline_segments');
        expect((await service.update(mapId, { ...payload, centerline_segments: [] })).centerline_segments).toEqual([]);
        expect((await service.get(mapId)).centerline_segments).toEqual([]);
    });

    it('loads and creates older maps without tags', async () => {
        const { centerline_segments, ...legacyPayload } = payload;
        stored = { _id: mapId, ...legacyPayload };
        expect((await service.get(mapId)).centerline_segments).toEqual([]);
        expect((await service.create(legacyPayload)).centerline_segments).toEqual([]);
    });

    it('groups legacy tags by range and persists segments on the next save', async () => {
        const { centerline_segments, ...legacyPayload } = payload;
        const legacyTags = [
            { id: 'speed', label: 'slow', start_position: 0.1, end_position: 0.25 },
            { id: 'turn-1', label: 'corner', start_position: 0.1, end_position: 0.25 },
            { id: 'duplicate-speed', label: 'slow', start_position: 0.1, end_position: 0.25 },
            { id: 'finish', label: 'corner', start_position: 0.95, end_position: 0.05 },
            { id: 'finish-speed', label: 'fast', start_position: 0.95, end_position: 0.05 },
        ];
        stored = new MapModel({ _id: mapId, ...legacyPayload, centerline_tags: legacyTags }).toObject();
        expect(stored.centerline_segments).toBeUndefined();
        const loaded = await service.get(mapId);
        expect(loaded.centerline_segments).toEqual([{ ...tag, tags: ['slow', 'corner'] }, wrapTag]);
        expect(loaded).not.toHaveProperty('centerline_tags');
        await service.update(mapId, { ...legacyPayload, centerline_segments: loaded.centerline_segments });
        expect(stored.centerline_segments).toEqual(loaded.centerline_segments);
        expect((await service.get(mapId)).centerline_segments).toEqual(loaded.centerline_segments);
        // Clearing migrated segments must take precedence over old stored tags.
        await service.update(mapId, { ...legacyPayload, centerline_segments: [] });
        expect((await service.get(mapId)).centerline_segments).toEqual([]);
    });

    it('accepts legacy clients while writing only the segment structure', async () => {
        const result = await service.create({ game: 'acc', circuit_name: 'Legacy', centerline_tags: [
            { id: 'turn-1', label: 'corner', start_position: 0.1, end_position: 0.25 },
            { id: 'speed', label: 'slow', start_position: 0.1, end_position: 0.25 },
        ] });
        expect(result.centerline_segments).toEqual([tag]);
        expect(stored.centerline_segments).toEqual([tag]);
        expect(stored.centerline_tags).toBeUndefined();
    });

    it.each([
        null, 'invalid', [null], [{ ...tag, tags: [] }], [{ ...tag, tags: 'corner' }],
        [{ ...tag, tags: ['corner', 1] }], [{ ...tag, tags: null }], [{ ...tag, id: '' }], [{ ...tag, tags: ['   '] }],
        [{ ...tag, tags: ['a'.repeat(121)] }], [{ ...tag, start_position: -0.1 }],
        [{ ...tag, end_position: 1.1 }], [{ ...tag, end_position: NaN }],
        [{ ...tag, start_position: '0.1' }], [{ ...tag, end_position: tag.start_position }],
        [tag, tag],
    ].map((centerline_segments) => ({ centerline_segments })))('rejects malformed tags before writing: $centerline_segments', async ({ centerline_segments }) => {
        const data = { ...payload, centerline_segments: centerline_segments as any };
        await expect(service.create(data)).rejects.toThrow(BadRequestException);
        await expect(service.update(mapId, data)).rejects.toThrow(BadRequestException);
        expect(storage.create).not.toHaveBeenCalled();
        expect(storage.findByIdAndUpdate).not.toHaveBeenCalled();
    });
});
