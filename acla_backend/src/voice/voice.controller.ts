import {
    Body,
    Controller,
    HttpCode,
    HttpStatus,
    Post,
    Res,
    UseGuards,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import type { Response } from 'express';
import {
    AiServiceClient,
    VoicePhonemesRequest,
    VoiceSynthesizeProsodyRequest,
} from '../shared/ai/ai-service.client';

@Controller('voice')
@UseGuards(AuthGuard('jwt'))
export class VoiceController {
    constructor(private readonly aiServiceClient: AiServiceClient) {}

    // The AI service validates both payloads against its phoneme/prosody schema.
    @Post('phonemes')
    @HttpCode(HttpStatus.OK)
    previewPhonemes(@Body() body: VoicePhonemesRequest) {
        return this.aiServiceClient.previewVoicePhonemes(body);
    }

    @Post('synthesize/prosody')
    @HttpCode(HttpStatus.OK)
    async synthesizeProsody(
        @Body() body: VoiceSynthesizeProsodyRequest,
        @Res() res: Response,
    ) {
        const wavBytes = await this.aiServiceClient.synthesizeVoiceProsody(body);
        res.set({
            'Content-Type': 'audio/wav',
            'Content-Length': wavBytes.length.toString(),
            'Content-Disposition': 'inline; filename="speech.wav"',
            'Cache-Control': 'no-store',
        });
        res.send(wavBytes);
    }
}
