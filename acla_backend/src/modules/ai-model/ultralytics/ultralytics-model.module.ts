import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import {
  UltralyticsModel,
  UltralyticsModelSchema,
} from '../../../schemas/ultralytics-model.schema';
import { GridFSModule } from '../../gridfs/gridfs.module';
import { UltralyticsModelService } from './ultralytics-model.service';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: UltralyticsModel.name, schema: UltralyticsModelSchema },
    ]),
    GridFSModule,
  ],
  providers: [UltralyticsModelService],
  exports: [UltralyticsModelService],
})
export class UltralyticsModelModule {}
