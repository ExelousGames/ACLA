import { Module } from '@nestjs/common';
import { ModelCommandProtocolController } from './model-command-protocol.controller';
import { VoiceGateway } from './voice.gateway';

@Module({
    controllers: [ModelCommandProtocolController],
    providers: [VoiceGateway],
})
export class VoiceModule {}
