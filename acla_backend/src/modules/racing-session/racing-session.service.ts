import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto } from 'src/dto/racing-session.dto';
import { RacingSession } from 'src/schemas/racing-session.schema';

@Injectable()
export class RacingSessionService {
    constructor(@InjectModel(RacingSession.name) private racingSession: Model<RacingSession>) {
    }

    //
    async retrieveAllRacingSessionsBasicInfo(mapName: string, userId: string): Promise<SessionBasicInfoListDto | null> {

        try {
            let racingMap: SessionBasicInfoListDto = new SessionBasicInfoListDto();

            const data = await this.racingSession.find({ 'map': mapName, 'user_id': userId }).select('session_name user_id').exec();
            data.forEach((element) => {
                racingMap.list.push({
                    name: element.session_name,
                    sessionId: element._id.toString()
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
            const data = await this.racingSession.findOne({ 'user_id': id }).exec();

            if (data) {
                session.session_name = data.session_name;
                session.map = data.map;
                session.userId = data.user_id.toString();
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
     * Creates a new racing session.
     * @param session_name 
     * @param map 
     * @param car_name 
     * @param userId 
     * @param data 
     * @returns 
     */
    async createRacingSession(session_name: string, map: string, car_name: string, userId: string, data: any[]) {
        return this.racingSession.create({
            session_name: session_name,
            map: map,
            car_name: car_name,
            user_id: userId,
            data: data
        });
    }

}
