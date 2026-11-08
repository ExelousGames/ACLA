import { Module } from '@nestjs/common';
import { AiServiceClient } from '../shared/ai/ai-service.client';
import { ModelCommandProtocolController } from './model-command-protocol.controller';
import { VoiceController } from './voice.controller';
import { VoiceGateway } from './voice.gateway';

@Module({
    controllers: [ModelCommandProtocolController, VoiceController],
    providers: [VoiceGateway, AiServiceClient],
})
export class VoiceModule {}
