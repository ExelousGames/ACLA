import { Module, forwardRef } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AiModelService } from './ai-model.service';
import { AiModelController } from './ai-model.controller';
import { AiServiceClient } from './ai-service.client';
import { AiModel, AiModelSchema } from 'src/schemas/ai-model.schema';
import { UserInfo, UserInfoSchema } from 'src/schemas/user-info.schema';
import { RacingSessionModule } from '../racing-session/racing-session.module';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: AiModel.name, schema: AiModelSchema },
            { name: UserInfo.name, schema: UserInfoSchema }
        ]),
        forwardRef(() => RacingSessionModule),
    ],
    controllers: [AiModelController],
    providers: [AiModelService, AiServiceClient],
    exports: [AiModelService, AiServiceClient],
})
export class AiModelModule { }
