import { Module } from '@nestjs/common';
import { AiService } from './ai-service.service';
import { AiController } from './ai-service.controller';
import { AiServiceClient } from './ai-service.client';

@Module({
    controllers: [AiController],
    providers: [AiService, AiServiceClient],
    exports: [AiService, AiServiceClient],
})
export class AiServiceModule { }
