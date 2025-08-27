import { Module } from '@nestjs/common';
import { RacingMapController } from './racing-map.controller';
import { RacingMapService } from './racing-map.service';
import { MongooseModule } from '@nestjs/mongoose';
import { RacingMap, RacingMapSchema } from 'src/schemas/map.schema';
import { AuthorizationModule } from '../../shared/authorization/authorization.module';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: RacingMap.name, schema: RacingMapSchema }]),
    AuthorizationModule
  ],
  controllers: [RacingMapController],
  providers: [RacingMapService],
  exports: [RacingMapService]
})
export class RacingMapModule { }
