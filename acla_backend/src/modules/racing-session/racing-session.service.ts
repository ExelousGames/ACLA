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
        return this.racingMap.findOne({ name: name }).exec();
    }

    async retrieveAllRacingSessionsInfo(): Promise<SessionBasicInfoListDto | null> {

        let racingMap: SessionBasicInfoListDto = new SessionBasicInfoListDto;
        this.racingMap.find().select('name').then(
            (data) => {

                data.forEach((element) => {
                    racingMap.list.push({
                        name: element.name,
                        map: element.map
                    });

                });
                return racingMap;
            }).catch((error) => {

            });

        return racingMap;
    }
}
