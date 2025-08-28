import { Controller, Post, Get, Body, Query, UseGuards, Request } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AiService } from './ai-service.service';

@Controller('ai')
export class AiController {
    constructor(private readonly aiService: AiService) { }

    @UseGuards(AuthGuard('jwt'))
    @Post('query')
    async processQuery(@Body() body: { question: string; dataset_id?: string }, @Request() req) {
        const query = {
            question: body.question,
            dataset_id: body.dataset_id,
            user_id: req.user?.email || req.user?.username,
            context: {
                user_id: req.user?.id
            }
        };

        return await this.aiService.processQuery(query);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload-dataset')
    async uploadDataset(@Body() dataset: any) {
        return await this.aiService.uploadDataset(dataset);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('analyze')
    async analyzeDataset(@Body() analysis: { dataset_id: string; analysis_type: string; parameters?: any }) {
        return await this.aiService.analyzeDataset(analysis);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('racing-session/ask')
    async askAboutRacingSession(
        @Body() body: { session_id: string; question: string },
        @Request() req
    ) {
        return await this.aiService.askQuestionAboutSession(
            body.session_id,
            body.question,
            req.user?.email || req.user?.username
        );
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('racing-session/insights')
    async getRacingSessionInsights(@Body() body: { session_id: string }) {
        return await this.aiService.getSessionInsights(body.session_id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('racing-session/patterns')
    async detectRacingPatterns(@Body() body: { session_id: string }) {
        return await this.aiService.detectRacingPatterns(body.session_id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('racing-session/performance-score')
    async getPerformanceScore(@Body() body: { session_id: string }) {
        return await this.aiService.getPerformanceScore(body.session_id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('racing-session/sector-analysis')
    async getSectorAnalysis(@Body() body: { session_id: string }) {
        return await this.aiService.getSectorAnalysis(body.session_id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('racing-session/optimal-prediction')
    async predictOptimalLapTime(@Body() body: { session_id: string }) {
        return await this.aiService.predictOptimalLapTime(body.session_id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Get('datasets')
    async listDatasets() {
        return await this.aiService.listDatasets();
    }

    @Get('health')
    async healthCheck() {
        return await this.aiService.healthCheck();
    }
}
