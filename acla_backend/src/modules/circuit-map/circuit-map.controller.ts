import { Body, Controller, Get, Param, Post, Put, Query, Request, UseGuards } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { CircuitMapGame } from 'src/schemas/circuit-map.schema';
import { CircuitMapService } from './circuit-map.service';

@Controller('circuit-map')
export class CircuitMapController {
    constructor(private readonly circuitMapService: CircuitMapService) { }

    @UseGuards(AuthGuard('jwt'))
    @Get('list')
    list(@Request() req, @Query('game') game?: CircuitMapGame) {
        return this.circuitMapService.list(game === 'other' ? 'other' : game === 'acc' ? 'acc' : undefined);
    }

    @UseGuards(AuthGuard('jwt'))
    @Get(':id')
    get(@Request() req, @Param('id') id: string) {
        return this.circuitMapService.get(id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post()
    create(@Request() req, @Body() body: any) {
        return this.circuitMapService.create(body);
    }

    @UseGuards(AuthGuard('jwt'))
    @Put(':id')
    update(@Request() req, @Param('id') id: string, @Body() body: any) {
        return this.circuitMapService.update(id, body);
    }
}
