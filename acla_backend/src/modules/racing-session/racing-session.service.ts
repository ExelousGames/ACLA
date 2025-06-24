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

        try {
            let racingMap: SessionBasicInfoListDto = new SessionBasicInfoListDto();

            const data = await this.racingSession.find({ 'map': mapName, 'user_email': username }).select('session_name id').exec();
            data.forEach((element) => {
                racingMap.list.push({
                    name: element.session_name,
                    id: element.id
                });
            });
            return racingMap;

        }
        catch (e) {
            // Handle errors appropriately
            throw new Error(`Failed to process data: ${e.message}`);
        }

    }

    async retrieveSessionDetailedInfo(id: string): Promise<RacingSessionDetailedInfoDto | null> {
        try {
            let session: RacingSessionDetailedInfoDto = new RacingSessionDetailedInfoDto;
            const data = await this.racingSession.findOne({ 'id': id }).exec();

            if (data) {
                session.session_name = data.session_name;
                session.id = data.id;
                session.map = data.map;
                session.user_email = data.user_email;
                session.points = data.points;
                session.data = data.data;

            }

            return session;
        } catch (error) {
            // Handle errors appropriately
            throw new Error(`Failed to process data: ${error.message}`);
        };
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
