import { BadRequestException } from '@nestjs/common';
import { model } from 'mongoose';
import { CircuitMapController } from './circuit-map.controller';
import { CircuitMapService } from './circuit-map.service';
import { CircuitMapSchema } from 'src/schemas/circuit-map.schema';

describe('Game-neutral circuit maps', () => {
    const id = '507f1f77bcf86cd799439011';
    const games = ['acc', 'ac', 'iracing', 'other', 'custom-simulator'];
    const payload = {
        game: 'custom-simulator',
        circuit_name: 'Test Circuit',
        source_track_key: 'track - layout',
        samples: { middle_line: [{
            bin: 250, normalized_position: 0.25, x: 10, y: 2, z: 30,
            sample_count: 2, updated_at: '2026-10-05T00:00:00.000Z',
        }] },
    };
    const MapModel = model('GameNeutralCircuitMap', CircuitMapSchema);

    it.each(games)('preserves %s metadata and coordinates on create and update', async (game) => {
        const data = { ...payload, game };
        const storage = {
            create: jest.fn(async (value) => ({ toObject: () => ({ _id: id, ...value }) })),
            findByIdAndUpdate: jest.fn((_id, value) => ({ lean: () => ({ exec: async () => ({ _id, ...value }) }) })),
        };
        const service = new CircuitMapService(storage as any);
        expect(await service.create(data)).toMatchObject({ ...data, sample_count: 1 });
        expect(await service.update(id, data)).toMatchObject({ ...data, sample_count: 1 });
        expect(storage.create).toHaveBeenCalledWith(expect.objectContaining({ game }));
        expect(storage.findByIdAndUpdate).toHaveBeenCalledWith(id, expect.objectContaining({ game }), { new: true });
    });

    it.each(games)('filters by the exact %s identifier from controller through storage', async (game) => {
        const data = { ...payload, game };
        const storage = { find: jest.fn(() => ({ sort: () => ({ lean: () => ({ exec: async () => [{ _id: id, ...data }] }) }) })) };
        const controller = new CircuitMapController(new CircuitMapService(storage as any));
        expect(await controller.list({}, game)).toMatchObject({ list: [{ id, game, source_track_key: payload.source_track_key }] });
        expect(storage.find).toHaveBeenCalledWith({ game });
    });

    it('lists all games when no filter is supplied', async () => {
        const storage = { find: jest.fn(() => ({ sort: () => ({ lean: () => ({ exec: async () => [] }) }) })) };
        const controller = new CircuitMapController(new CircuitMapService(storage as any));
        await expect(controller.list({})).resolves.toEqual({ list: [] });
        expect(storage.find).toHaveBeenCalledWith({});
    });

    it.each(games)('validates and preserves %s in the persisted schema', (game) => {
        const document = new MapModel({ ...payload, game });
        expect(document.validateSync()).toBeUndefined();
        expect(document.game).toBe(game);
    });

    it('requires an explicit game instead of assigning a default', async () => {
        const { game, ...withoutGame } = payload;
        const storage = { create: jest.fn() };
        await expect(new CircuitMapService(storage as any).create(withoutGame)).rejects.toThrow(BadRequestException);
        expect(storage.create).not.toHaveBeenCalled();
        const document = new MapModel(withoutGame);
        expect(document.game).toBeUndefined();
        expect(document.validateSync()?.errors.game).toBeDefined();
    });

    it('preserves the saved game when an update omits it', async () => {
        const { game, ...withoutGame } = payload;
        const storage = {
            findByIdAndUpdate: jest.fn((_id, data) => ({ lean: () => ({ exec: async () => ({ _id, ...payload, ...data }) }) })),
        };
        const result = await new CircuitMapService(storage as any).update(id, withoutGame);
        expect(result.game).toBe(game);
        expect(storage.findByIdAndUpdate.mock.calls[0][1]).not.toHaveProperty('game');
    });

    it.each([null, '', '   ', 42, false, [], { $ne: null }])('rejects invalid game metadata and filters %p before accessing storage', async (game) => {
        const storage = { create: jest.fn(), findByIdAndUpdate: jest.fn(), find: jest.fn() };
        const service = new CircuitMapService(storage as any);
        const data = { ...payload, game: game as any };
        await expect(service.create(data)).rejects.toThrow(BadRequestException);
        await expect(service.update(id, data)).rejects.toThrow(BadRequestException);
        await expect(new CircuitMapController(service).list({}, game as any)).rejects.toThrow(BadRequestException);
        expect(storage.create).not.toHaveBeenCalled();
        expect(storage.findByIdAndUpdate).not.toHaveBeenCalled();
        expect(storage.find).not.toHaveBeenCalled();
    });
});
