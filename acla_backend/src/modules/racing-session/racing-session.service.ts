import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto } from 'src/dto/racing-session.dto';
import { RacingSession } from 'src/schemas/racing-session.schema';

@Injectable()
export class RacingSessionService {
    constructor(@InjectModel(RacingSession.name) private racingSession: Model<RacingSession>) {
    }

    async retrieveAllRacingSessionsInfo(mapName: string, username: string): Promise<SessionBasicInfoListDto | null> {

        let racingMap: SessionBasicInfoListDto = new SessionBasicInfoListDto;
        this.racingSession.find().select({ 'map': mapName, 'user_email': username }).then(
            (data) => {

                data.forEach((element) => {
                    racingMap.list.push({
                        name: element.session_name,
                    });

                });
                return racingMap;
            }).catch((error) => {

            });

        return racingMap;
    }

    async retrieveSessionDetailedInfo(mapName: string, session_name: string, username: string): Promise<RacingSessionDetailedInfoDto | null> {

        let racingMap: RacingSessionDetailedInfoDto = new RacingSessionDetailedInfoDto;
        this.racingSession.find().select({ 'map': mapName, 'session_name': session_name, 'user_email': username }).then(
            (data) => {

                data.forEach((element) => {
                    racingMap.list.push({
                        name: element.session_name,
                    });

                });
                return racingMap;
            }).catch((error) => {

            });

        return racingMap;
    }


    /**
     * 
     * @param session_name 
     * @param id 
     * @param map 
     * @param user_email 
     * @param data 
     * @returns 
     */
    async createRacingSession(session_name: string, id: string, map: string, user_email: string, data: any[]) {
        return this.racingSession.create({
            session_name,
            id,
            map,
            user_email,
            data
        });
    }

}
