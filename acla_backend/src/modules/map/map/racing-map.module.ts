import { Module } from '@nestjs/common';
import { RacingMapController } from './racing-map.controller';
import { RacingMapService } from './racing-map.service';
import { MongooseModule } from '@nestjs/mongoose';
import { RacingMap, RacingMapSchema } from 'src/schemas/map.schema';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: RacingMap.name, schema: RacingMapSchema }]),
  ],
  controllers: [RacingMapController],
  providers: [RacingMapService],
  exports: [RacingMapService]
})
export class RacingMapModule { }
