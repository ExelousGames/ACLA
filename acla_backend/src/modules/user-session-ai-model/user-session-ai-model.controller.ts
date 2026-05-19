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

    @UseGuards(AuthGuard('jwt'))
    @Post('ai-query')
    async processAIQuery(
        @Body() body: {
            question: string;
            context?: any;
        },
        @Request() req: any
    ) {
        // Unified endpoint for AI queries - handles both model operations and model-related questions
        // Examples:
        // - "Train a new model for Monza" (ai_model_operation)
        // - "Which of my models performs best for Monza?" (model_query)
        // - General AI queries (general)

        const queryRequest = {
            question: body.question,
            user_id: req?.user?.id,
            context: {
                ...body.context,
            }
        };

        return this.aiModelService.processAIQuery(queryRequest);
    }

    /**
     * Phase 2.5 — Streaming variant of /ai-query.
     * Pipes the AI service's Server-Sent Events directly to the client.
     * The frontend renders tokens as they arrive and queues audio events
     * for gapless sentence-by-sentence playback.
     */
    @UseGuards(AuthGuard('jwt'))
    @Post('ai-query/stream')
    async streamAIQuery(
        @Body() body: { question: string; context?: any },
        @Request() req: any,
        @Res() res: Response,
    ) {
        const queryRequest = {
            question: body.question,
            user_id: req?.user?.id,
            context: { ...body.context },
        };

        const upstream = await this.aiModelService.streamAIQuery(queryRequest);

        // SSE response headers — must be set BEFORE piping any data.
        res.set({
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        });
        res.flushHeaders?.();

        // Pipe upstream bytes straight through. The Python service already
        // formats them as SSE; we don't re-parse, just forward.
        upstream.pipe(res);

        // Clean up if the client disconnects mid-stream.
        const cleanup = () => {
            try { (upstream as any).destroy?.(); } catch { /* ignore */ }
        };
        res.on('close', cleanup);
        res.on('error', cleanup);
        upstream.on('error', (err) => {
            console.error('[ai-query/stream] upstream error:', err.message);
            cleanup();
            if (!res.writableEnded) res.end();
        });
        upstream.on('end', () => {
            if (!res.writableEnded) res.end();
        });
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
