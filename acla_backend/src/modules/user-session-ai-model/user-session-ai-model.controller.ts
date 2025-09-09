import { Controller, Get, Post, Put, Delete, Body, Param, Query, UseGuards, Request, Inject, forwardRef } from '@nestjs/common';
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
            sessionId?: string;
            trackName?: string;
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
                track_name: body.trackName,
            }
        };

        return this.aiModelService.processAIQuery(queryRequest);
    }

    @Get('health')
    async healthCheck() {
        return await this.aiModelService.healthCheck();
    }
}
