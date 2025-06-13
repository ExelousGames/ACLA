import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { AllMapsBasicInfoListDto, MapBasicInfo } from 'src/dto/map.dto';
import { RacingMap } from 'src/schemas/map.schema';

@Injectable()
export class RacingMapService {

    constructor(@InjectModel(RacingMap.name) private racingMap: Model<RacingMap>) {
    }

    async getRacingMap(name: string): Promise<RacingMap | null> {
        return this.racingMap.findOne({ name: name }).exec();
    }

    async retrieveAllMapBasicInfos(): Promise<AllMapsBasicInfoListDto | null> {

        try {
            let racingMap = new AllMapsBasicInfoListDto();
            const rawData = await this.racingMap.find().exec();
            racingMap.list = rawData.map((element) => {

                return { name: element.name };

            });

            return racingMap;
        }
        catch (error) {
            // Handle errors appropriately
            throw new Error(`Failed to process data: ${error.message}`);
        }

    }
}
