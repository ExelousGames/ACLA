import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { AllMapsBasicInfoListDto } from 'src/dto/map.dto';
import { RacingMap } from 'src/schemas/map.schema';

@Injectable()
export class RacingMapService {

    constructor(@InjectModel(RacingMap.name) private racingMap: Model<RacingMap>) {
    }

    async getRacingMap(name: string): Promise<RacingMap | null> {
        return this.racingMap.findOne({ name: name }).exec();
    }

    async retrieveAllMapBasicInfos(): Promise<AllMapsBasicInfoListDto | null> {

        let racingMap: AllMapsBasicInfoListDto = new AllMapsBasicInfoListDto;
        this.racingMap.find().then((data) => {

        })
        this.racingMap.find().select('name').then(
            (data) => {

                data.forEach((element) => {
                    racingMap.list.push({
                        name: element.name
                    });

                });
                return racingMap;
            }).catch((error) => {

            });

        return racingMap;
    }
}
