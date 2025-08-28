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
    @Post('predict')
    async makePrediction(@Body() modelPredictionDto: ModelPredictionDto) {
        return this.aiModelService.makePrediction(modelPredictionDto);
    }

    @UseGuards(AuthGuard('jwt'))
    @Get('performance-history/:trackName/:modelType')
    async getPerformanceHistory(
        @Param('trackName') trackName: string,
        @Param('modelType') modelType: string,
        @Request() req: any
    ) {
        return this.aiModelService.getModelPerformanceHistory(req?.user?.id || '', trackName, modelType);
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
            query: string;
            sessionId?: string;
            context?: any;
        },
        @Request() req: any
    ) {
        // Forward natural language queries to the AI service
        const queryRequest = {
            question: body.query,
            dataset_id: body.sessionId,
            user_id: req?.user?.id,
            context: {
                ...body.context,
                user_id: req?.user?.id,
                type: 'ai_model_operation'
            }
        };

        return this.aiModelService.processAIQuery(queryRequest);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('intelligent-training')
    async intelligentTraining(
        @Body() body: {
            naturalLanguageRequest: string;
            context?: any;
        },
        @Request() req: any
    ) {
        // Allow users to request model training using natural language
        // e.g., "Train a lap time prediction model for Spa using my last 5 sessions"
        const queryRequest = {
            question: body.naturalLanguageRequest,
            user_id: req?.user?.id,
            context: {
                ...body.context,
                user_id: req?.user?.id,
                type: 'model_training',
                operation: 'train'
            }
        };

        return this.aiModelService.processAIQuery(queryRequest);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('ask-about-models')
    async askAboutModels(
        @Body() body: {
            question: string;
            trackName?: string;
            modelType?: string;
        },
        @Request() req: any
    ) {
        // Allow users to ask questions about their models
        // e.g., "Which of my models performs best for Monza?"
        const queryRequest = {
            question: body.question,
            user_id: req?.user?.id,
            context: {
                user_id: req?.user?.id,
                track_name: body.trackName,
                model_type: body.modelType,
                type: 'model_query'
            }
        };

        return this.aiModelService.processAIQuery(queryRequest);
    }
}
