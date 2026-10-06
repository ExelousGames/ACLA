import { BadRequestException } from '@nestjs/common';
import { model } from 'mongoose';
import { CircuitMapSchema } from 'src/schemas/circuit-map.schema';
import { CircuitMapService } from './circuit-map.service';

describe('Circuit map centerline tags', () => {
    const mapId = '507f1f77bcf86cd799439011';
    const MapModel = model('CircuitMapTags', CircuitMapSchema);
    const tag = { id: 'turn-1', label: 'Turn 1', start_position: 0.1, end_position: 0.25 };
    const wrapTag = { id: 'finish', label: 'Across the line', start_position: 0.95, end_position: 0.05 };
    const payload = {
        game: 'acc' as const, circuit_name: 'Tagged Circuit', centerline_tags: [tag, wrapTag],
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
        expect((await service.create(payload)).centerline_tags).toEqual([tag, wrapTag]);
        expect((await service.get(mapId)).centerline_tags).toEqual([tag, wrapTag]);
        const updated = { ...tag, label: '  Braking zone  ' };
        const result = await service.update(mapId, { ...payload, centerline_tags: [updated] });
        expect(result.centerline_tags).toEqual([{ ...tag, label: 'Braking zone' }]);
        expect((await service.get(mapId)).centerline_tags).toEqual(result.centerline_tags);
        expect(result.samples.middle_line).toEqual(payload.samples.middle_line);
    });

    it('clears removed tags and preserves tags when an older client omits the field', async () => {
        const { centerline_tags, ...legacyPayload } = payload;
        expect((await service.update(mapId, legacyPayload)).centerline_tags).toEqual(centerline_tags);
        expect(storage.findByIdAndUpdate.mock.calls[0][1]).not.toHaveProperty('centerline_tags');
        expect((await service.update(mapId, { ...payload, centerline_tags: [] })).centerline_tags).toEqual([]);
        expect((await service.get(mapId)).centerline_tags).toEqual([]);
    });

    it('loads and creates older maps without tags', async () => {
        const { centerline_tags, ...legacyPayload } = payload;
        stored = { _id: mapId, ...legacyPayload };
        expect((await service.get(mapId)).centerline_tags).toEqual([]);
        expect((await service.create(legacyPayload)).centerline_tags).toEqual([]);
    });

    it.each([
        null, 'invalid', [null], [{ ...tag, id: '' }], [{ ...tag, label: '   ' }],
        [{ ...tag, label: 'a'.repeat(121) }], [{ ...tag, start_position: -0.1 }],
        [{ ...tag, end_position: 1.1 }], [{ ...tag, end_position: NaN }],
        [{ ...tag, start_position: '0.1' }], [{ ...tag, end_position: tag.start_position }],
        [tag, tag],
    ].map((centerline_tags) => ({ centerline_tags })))('rejects malformed tags before writing: $centerline_tags', async ({ centerline_tags }) => {
        const data = { ...payload, centerline_tags: centerline_tags as any };
        await expect(service.create(data)).rejects.toThrow(BadRequestException);
        await expect(service.update(mapId, data)).rejects.toThrow(BadRequestException);
        expect(storage.create).not.toHaveBeenCalled();
        expect(storage.findByIdAndUpdate).not.toHaveBeenCalled();
    });
});
