import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AiModelController } from './ai-model.controller';
import { AiModelService } from './ai-model.service';
import { AIModel, AIModelSchema } from '../../schemas/ai-model.schema';
import { GridFSModule } from '../gridfs/gridfs.module';
import { UltralyticsModelModule } from './ultralytics/ultralytics-model.module';
import { UltralyticsModelController } from './ultralytics/ultralytics-model.controller';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: AIModel.name, schema: AIModelSchema },
        ]),
        GridFSModule,
        UltralyticsModelModule,
    ],
    // Register model-specific routes before the generic :id route.
    controllers: [UltralyticsModelController, AiModelController],
    providers: [AiModelService],
    exports: [AiModelService],
})
export class AiModelModule { }
