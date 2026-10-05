import 'reflect-metadata';
import { HttpStatus, RequestMethod } from '@nestjs/common';
import { GUARDS_METADATA, HTTP_CODE_METADATA, METHOD_METADATA, PATH_METADATA } from '@nestjs/common/constants';
import { AuthGuard } from '@nestjs/passport';
import { CircuitMapController } from './circuit-map.controller';
import { CircuitMapService } from './circuit-map.service';

describe('CircuitMapController removal', () => {
    it('exposes an authenticated DELETE endpoint returning no content', () => {
        const handler = CircuitMapController.prototype.remove;
        expect(Reflect.getMetadata(PATH_METADATA, CircuitMapController)).toBe('circuit-map');
        expect(Reflect.getMetadata(PATH_METADATA, handler)).toBe(':id');
        expect(Reflect.getMetadata(METHOD_METADATA, handler)).toBe(RequestMethod.DELETE);
        expect(Reflect.getMetadata(HTTP_CODE_METADATA, handler)).toBe(HttpStatus.NO_CONTENT);
        expect(Reflect.getMetadata(GUARDS_METADATA, handler)).toContain(AuthGuard('jwt'));
    });

    it('passes the requested map id to the service', async () => {
        const service = { remove: jest.fn().mockResolvedValue(undefined) };
        const controller = new CircuitMapController(service as unknown as CircuitMapService);
        await expect(controller.remove('507f1f77bcf86cd799439011')).resolves.toBeUndefined();
        expect(service.remove).toHaveBeenCalledWith('507f1f77bcf86cd799439011');
    });
});
