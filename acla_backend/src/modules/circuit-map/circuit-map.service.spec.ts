import { BadRequestException, NotFoundException } from '@nestjs/common';
import { CircuitMapService } from './circuit-map.service';

describe('CircuitMapService middle line capture', () => {
    const mapId = '507f1f77bcf86cd799439011';
    const sample = {
        bin: 420, normalized_position: 0.42, x: 12, y: 0, z: 34,
        sample_count: 3, updated_at: '2026-01-01T00:00:00.000Z', locked: true,
    };
    const payload = {
        game: 'custom-simulator',
        circuit_name: 'Middle Test Circuit',
        samples: { middle_line: [sample], left_boundary: [{ ...sample, x: 2 }] },
    };
    let service: CircuitMapService;
    let model: { create: jest.Mock; findById: jest.Mock; findByIdAndUpdate: jest.Mock };

    beforeEach(() => {
        model = {
            create: jest.fn(async (data) => ({ toObject: () => ({ _id: mapId, ...data }) })),
            findById: jest.fn().mockReturnValue({
                lean: () => ({ exec: async () => ({ _id: mapId, ...payload }) }),
            }),
            findByIdAndUpdate: jest.fn((_id, data) => ({
                lean: () => ({ exec: async () => ({ _id, ...data }) }),
            })),
        };
        service = new CircuitMapService(model as any);
    });

    it('persists middle line samples on create and includes them in the sample count', async () => {
        const result = await service.create(payload);
        expect(model.create).toHaveBeenCalledWith(expect.objectContaining({
            samples: { ...payload.samples, right_boundary: [], pit_lane: [] },
            sample_count: 2,
        }));
        expect(result.samples.middle_line).toEqual([sample]);
        expect(result.sample_count).toBe(2);
    });

    it('updates middle line samples alongside other captured paths', async () => {
        const result = await service.update(mapId, payload);
        expect(model.findByIdAndUpdate).toHaveBeenCalledWith(mapId, expect.objectContaining({
            samples: { ...payload.samples, right_boundary: [], pit_lane: [] },
            sample_count: 2,
        }), { new: true });
        expect(result.samples.middle_line).toEqual([sample]);
        expect(result.sample_count).toBe(2);
    });

    it('returns stored middle line samples and counts them when no count is stored', async () => {
        const result = await service.get(mapId);
        expect(result.samples.middle_line).toEqual([sample]);
        expect(result.samples.left_boundary).toEqual(payload.samples.left_boundary);
        expect(result.sample_count).toBe(2);
    });

    it('keeps older maps without a middle line compatible', async () => {
        model.findById.mockReturnValue({
            lean: () => ({ exec: async () => ({
                _id: mapId, circuit_name: 'Old Circuit', samples: { left_boundary: [sample] },
            }) }),
        });
        const result = await service.get(mapId);
        expect(result.samples.middle_line).toEqual([]);
        expect(result.samples.left_boundary).toEqual([sample]);
        expect(result.sample_count).toBe(1);
    });
});

describe('CircuitMapService removal', () => {
    const mapId = '507f1f77bcf86cd799439011';
    let service: CircuitMapService;
    let exec: jest.Mock;
    let model: { deleteOne: jest.Mock };

    beforeEach(() => {
        exec = jest.fn().mockResolvedValue({ deletedCount: 1 });
        model = { deleteOne: jest.fn().mockReturnValue({ exec }) };
        service = new CircuitMapService(model as any);
    });

    it('removes only the requested global map, including its embedded samples', async () => {
        await expect(service.remove(mapId)).resolves.toBeUndefined();
        expect(model.deleteOne).toHaveBeenCalledWith({ _id: mapId });
        expect(exec).toHaveBeenCalledTimes(1);
    });

    it('rejects an invalid id before accessing storage', async () => {
        await expect(service.remove('invalid')).rejects.toBeInstanceOf(BadRequestException);
        expect(model.deleteOne).not.toHaveBeenCalled();
    });

    it('reports a missing map', async () => {
        exec.mockResolvedValue({ deletedCount: 0 });
        await expect(service.remove(mapId)).rejects.toThrow(new NotFoundException('Circuit map not found'));
    });

    it('propagates storage failures', async () => {
        exec.mockRejectedValue(new Error('Storage unavailable'));
        await expect(service.remove(mapId)).rejects.toThrow('Storage unavailable');
    });
});
