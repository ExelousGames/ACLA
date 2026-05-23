import { Controller, Get, Post, Put, Delete, Body, Param, Query, UseGuards, Request, Inject, forwardRef, Res, HttpException, HttpStatus } from '@nestjs/common';
import type { Response } from 'express';
import { AuthGuard } from '@nestjs/passport';
import { Types } from 'mongoose';
import { UserSessionAiModelService } from './user-session-ai-model.service';
import { CreateSeesionAIModelDto, UpdateAiModelDto, GetAiModelDto, IncrementalTrainingDto, ModelPredictionDto } from 'src/dto/ai-model.dto';

@Controller('user-ai-model')
export class UserSessionAiModelController {
    constructor(private readonly aiModelService: UserSessionAiModelService) { }

    @UseGuards(AuthGuard('jwt'))
    @Post()
    async createModel(@Body() createAiModelDto: CreateSeesionAIModelDto, @Request() req: any) {
        // Ensure the user can only create models for themselves
        createAiModelDto.userId = req?.user?.id || '';
        return this.aiModelService.createModel(createAiModelDto);
    }

    @UseGuards(AuthGuard('jwt'))
    @Get('user/:trackName')
    async getUserModels(
        @Param('trackName') trackName: string,
        @Query('modelType') modelType?: string,
        @Query('activeOnly') activeOnly?: string,
        @Request() req?: any
    ) {
        const getUserDto: GetAiModelDto = {
            userId: req?.user?.id || '',
            trackName: trackName,
            modelType: modelType,
            activeOnly: activeOnly === 'true',
        };
        return this.aiModelService.findModelsByUser(getUserDto);
    }

    @UseGuards(AuthGuard('jwt'))
    @Get('active/:trackName/:modelType')
    async getActiveModel(
        @Param('trackName') trackName: string,
        @Param('carName') carName: string,
        @Param('modelType') modelType: string,
        @Param('target_variable') target_variable: string,
        @Request() req: any
    ) {
        return this.aiModelService.findActiveUserSessionAIModel(req?.user?.id || '', trackName, carName, modelType, target_variable);
    }

    @UseGuards(AuthGuard('jwt'))
    @Put(':id')
    async updateModel(
        @Param('id') id: string,
        @Body() updateAiModelDto: UpdateAiModelDto
    ) {
        return this.aiModelService.updateModel(id, updateAiModelDto);
    }

    @UseGuards(AuthGuard('jwt'))
    @Delete(':id')
    async deleteModel(@Param('id') id: string) {
        return this.aiModelService.deleteModel(id);
    }

    @Get('health')
    async healthCheck() {
        return await this.aiModelService.healthCheck();
    }

    /**
     * Phase 2 — Neural text-to-speech (Kokoro) for AI chat responses.
     * Replaces the browser's robotic window.speechSynthesis.
     *
     * Auth-protected: only logged-in users hit the AI service's TTS.
     * Returns audio/wav bytes that the renderer plays via HTMLAudioElement.
     */
    @UseGuards(AuthGuard('jwt'))
    @Post('voice-synthesize')
    async voiceSynthesize(
        @Body() body: {
            text: string;
            voice?: string;
            speed?: number;
            language?: string;
        },
        @Res() res: Response,
    ) {
        if (!body?.text || !body.text.trim()) {
            throw new HttpException(
                'voice-synthesize: "text" is required',
                HttpStatus.BAD_REQUEST,
            );
        }

        const wavBytes = await this.aiModelService.synthesizeVoice({
            text: body.text,
            voice: body.voice,
            speed: body.speed,
            language: body.language,
        });

        res.set({
            'Content-Type': 'audio/wav',
            'Content-Length': wavBytes.length.toString(),
            'Cache-Control': 'no-store',
        });
        res.send(wavBytes);
    }

    /** Phase 2 — list available Kokoro voices (for a future voice picker UI). */
    @UseGuards(AuthGuard('jwt'))
    @Get('voices')
    async listVoices() {
        return await this.aiModelService.listVoices();
    }
}
