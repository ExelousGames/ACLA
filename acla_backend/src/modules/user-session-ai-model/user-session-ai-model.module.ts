import { Module, forwardRef } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { UserSessionAiModelService } from './user-session-ai-model.service';
import { UserSessionAiModelController } from './user-session-ai-model.controller';
import { AiServiceClient } from '../../shared/ai/ai-service.client';
import { UserACCTrackAIModel, SessionAIModelSchema } from 'src/schemas/session-ai-model.schema';
import { UserInfo, UserInfoSchema } from 'src/schemas/user-info.schema';
import { RacingSessionModule } from '../racing-session/racing-session.module';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: UserACCTrackAIModel.name, schema: SessionAIModelSchema },
            { name: UserInfo.name, schema: UserInfoSchema }
        ]),
        forwardRef(() => RacingSessionModule),
    ],
    controllers: [UserSessionAiModelController],
    providers: [UserSessionAiModelService, AiServiceClient],
    exports: [UserSessionAiModelService, AiServiceClient],
})
export class AiModelModule { }
