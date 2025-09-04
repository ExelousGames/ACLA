import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AiModelController } from './ai-model.controller';
import { AiModelService } from './ai-model.service';
import { AIModel, AIModelSchema } from '../../schemas/ai-model.schema';
import { GridFSModule } from '../gridfs/gridfs.module';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: AIModel.name, schema: AIModelSchema },
        ]),
        GridFSModule,
    ],
    controllers: [AiModelController],
    providers: [AiModelService],
    exports: [AiModelService],
})
export class AiModelModule { }