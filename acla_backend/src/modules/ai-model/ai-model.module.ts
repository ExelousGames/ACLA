import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AiModelController } from './ai-model.controller';
import { AiModelService } from './ai-model.service';
import { AIModel, AIModelSchema } from '../../schemas/ai-model.schema';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: AIModel.name, schema: AIModelSchema },
        ]),
    ],
    controllers: [AiModelController],
    providers: [AiModelService],
    exports: [AiModelService],
})
export class AiModelModule { }