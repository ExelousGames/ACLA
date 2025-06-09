import { Controller, Get, UseGuards, Request } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AllMapsBasicInfoListDto } from 'src/dto/map.dto';
import { RacingMapService } from './racing-map.service';

@Controller('racing_map')
export class RacingMapController {

    constructor(private racingMapService: RacingMapService) { }

    @UseGuards(AuthGuard('jwt'))
    @Get('allmapbasicinfos')
    getAllMapsBasicInfos(@Request() req): AllMapsBasicInfoListDto {
        this.racingMapService.retrieveAllMapBasicInfos();
        return req.user;
    }
}
