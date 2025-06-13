import { Controller, Get, UseGuards, Request, Body, Post } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AllMapsBasicInfoListDto } from 'src/dto/map.dto';
import { RacingMapService } from './racing-map.service';
import { RacingMap } from 'src/schemas/map.schema';

@Controller('racingmap')
export class RacingMapController {

    constructor(private racingMapService: RacingMapService) { }

    @UseGuards(AuthGuard('jwt'))
    @Get('map/infolists')
    async getAllMapsBasicInfos(@Body() body: any): Promise<AllMapsBasicInfoListDto | null> {
        return this.racingMapService.retrieveAllMapBasicInfos();
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('map/infolists')
    async getMap(@Body() body: any): Promise<RacingMap | null> {
        console.log(body.name)
        return this.racingMapService.getRacingMap(body.name);
    }
}
