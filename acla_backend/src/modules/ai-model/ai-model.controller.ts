import { Controller, Get, Post, Put, Delete, Body, Param, Query, UseGuards, Request, Inject, forwardRef } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { Types } from 'mongoose';
import { AiModelService } from './ai-model.service';
import { CreateAiModelDto, UpdateAiModelDto, GetAiModelDto, IncrementalTrainingDto, ModelPredictionDto } from 'src/dto/ai-model.dto';

@Controller('ai-model')
export class AiModelController {
    constructor(private readonly aiModelService: AiModelService) { }

    @UseGuards(AuthGuard('jwt'))
    @Post()
    async createModel(@Body() createAiModelDto: CreateAiModelDto, @Request() req: any) {
        // Ensure the user can only create models for themselves
        createAiModelDto.userId = new Types.ObjectId(req?.user?.id || '');
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
            userId: new Types.ObjectId(req?.user?.id || ''),
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
        @Param('modelType') modelType: string,
        @Request() req: any
    ) {
        return this.aiModelService.findActiveModel(req?.user?.id || '', trackName, modelType);
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
    @Post('incremental-training')
    async performIncrementalTraining(@Body() incrementalTrainingDto: IncrementalTrainingDto) {
        return this.aiModelService.incrementalTraining(incrementalTrainingDto);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('train-new')
    async trainNewModel(
        @Body() body: {
            trackName: string;
            modelType: string;
            modelName: string;
            sessionIds: string[];
            trainingParameters?: any;
        },
        @Request() req: any
    ) {
        // This endpoint will create a new model from scratch using specified racing sessions
        const createDto: CreateAiModelDto = {
            userId: new Types.ObjectId(req?.user?.id || ''),
            trackName: body.trackName,
            modelName: body.modelName,
            modelVersion: '1.0.0',
            modelData: {}, // Will be populated by AI service
            modelMetadata: {
                trainingSessionsCount: body.sessionIds.length,
                lastTrainingDate: new Date(),
                performanceMetrics: {},
                modelType: body.modelType,
                features: [],
            },
            trainingSessionIds: body.sessionIds,
            isActive: true,
            description: `Initial model trained on ${body.sessionIds.length} sessions`,
        };

        // Here you would call the AI service to actually train the model
        // For now, we'll create the model entry and let the AI service populate it
        return this.aiModelService.createModel(createDto);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('ai-query')
    async processAIQuery(
        @Body() body: {
            question: string;
            sessionId?: string;
            trackName?: string;
            modelType?: string;
            queryType?: 'ai_model_operation' | 'model_query' | 'general';
            context?: any;
        },
        @Request() req: any
    ) {
        // Unified endpoint for AI queries - handles both model operations and model-related questions
        // Examples:
        // - "Train a new model for Monza" (ai_model_operation)
        // - "Which of my models performs best for Monza?" (model_query)
        // - General AI queries (general)

        const queryType = body.queryType || (body.trackName || body.modelType ? 'model_query' : 'ai_model_operation');

        const queryRequest = {
            question: body.question,
            dataset_id: body.sessionId,
            user_id: req?.user?.id,
            context: {
                ...body.context,
                user_id: req?.user?.id,
                track_name: body.trackName,
                model_type: body.modelType,
                type: queryType
            }
        };

        return this.aiModelService.processAIQuery(queryRequest);
    }

    @Get('health')
    async healthCheck() {
        return await this.aiModelService.healthCheck();
    }
}
