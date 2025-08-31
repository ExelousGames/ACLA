import { Module, forwardRef } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AiModelService } from './ai-model.service';
import { AiModelController } from './ai-model.controller';
import { AiServiceClient } from './ai-service.client';
import { UserTrackAIModel, SessionAIModelSchema } from 'src/schemas/session-ai-model.schema';
import { UserInfo, UserInfoSchema } from 'src/schemas/user-info.schema';
import { RacingSessionModule } from '../racing-session/racing-session.module';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: UserTrackAIModel.name, schema: SessionAIModelSchema },
            { name: UserInfo.name, schema: UserInfoSchema }
        ]),
        forwardRef(() => RacingSessionModule),
    ],
    controllers: [AiModelController],
    providers: [AiModelService, AiServiceClient],
    exports: [AiModelService, AiServiceClient],
})
export class AiModelModule { }
