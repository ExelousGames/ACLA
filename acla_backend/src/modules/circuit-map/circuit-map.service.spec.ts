import { BadRequestException, NotFoundException } from '@nestjs/common';
import { CircuitMapService } from './circuit-map.service';

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
