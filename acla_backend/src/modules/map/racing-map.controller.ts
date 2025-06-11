import { Controller, Get, UseGuards, Request, Body } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AllMapsBasicInfoListDto } from 'src/dto/map.dto';
import { RacingMapService } from './racing-map.service';

@Controller('racingmap')
export class RacingMapController {

    constructor(private racingMapService: RacingMapService) { }

    @UseGuards(AuthGuard('jwt'))
    @Get('allmapbasicinfos')
    async getAllMapsBasicInfos(@Body() body: any): Promise<AllMapsBasicInfoListDto | null> {
        console.log("here");
        return this.racingMapService.retrieveAllMapBasicInfos();
    }
}
