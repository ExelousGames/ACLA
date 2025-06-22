import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { SessionBasicInfoListDto } from 'src/dto/racing-session.dto';
import { RacingSession } from 'src/schemas/racing-session.schema';

@Injectable()
export class RacingSessionService {
    constructor(@InjectModel(RacingSession.name) private racingMap: Model<RacingSession>) {
    }

    async getRacingMap(name: string): Promise<RacingSession | null> {
        return this.racingMap.findOne({ session_name: name }).exec();
    }

    async retrieveAllRacingSessionsInfo(): Promise<SessionBasicInfoListDto | null> {

        let racingMap: SessionBasicInfoListDto = new SessionBasicInfoListDto;
        this.racingMap.find().select('name').then(
            (data) => {

                data.forEach((element) => {
                    racingMap.list.push({
                        name: element.session_name,
                        map: element.map
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
        return this.racingMap.create({
            session_name,
            id,
            map,
            user_email,
            data
        });
    }

}
