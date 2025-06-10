import { Controller, Get, UseGuards, Request } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { SessionBasicInfoListDto } from 'src/dto/racing-session.dto';
import { RacingSessionService } from './racing-session.service';

@Controller('racing-session')
export class RacingSessionController {
    constructor(private racingSessionService: RacingSessionService) { }

    @UseGuards(AuthGuard('jwt'))
    @Get('allmapbasicinfos')
    retrieveAllRacingSessionsInfo(@Request() req): SessionBasicInfoListDto {
        this.racingSessionService.retrieveAllRacingSessionsInfo();
        return req.user;
    }

}
