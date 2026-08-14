import { Module } from '@nestjs/common';
import { SessionToolsController } from './session-tools.controller';
import { VoiceGateway } from './voice.gateway';

@Module({
    controllers: [SessionToolsController],
    providers: [VoiceGateway],
})
export class VoiceModule {}
