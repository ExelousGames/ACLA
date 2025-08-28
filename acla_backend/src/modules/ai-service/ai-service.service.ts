import { Injectable } from '@nestjs/common';
import { AiServiceClient, QueryRequest, AnalysisRequest, DatasetUploadRequest } from './ai-service.client';

@Injectable()
export class AiService {
    constructor(private readonly aiServiceClient: AiServiceClient) { }

    async uploadDataset(dataset: DatasetUploadRequest) {
        return await this.aiServiceClient.uploadDataset(dataset);
    }

    async processQuery(query: QueryRequest) {
        return await this.aiServiceClient.processQuery(query);
    }

    async analyzeDataset(analysis: AnalysisRequest) {
        return await this.aiServiceClient.analyzeDataset(analysis);
    }

    async analyzeRacingSession(sessionData: any) {
        return await this.aiServiceClient.analyzeRacingSession(sessionData);
    }

    async listDatasets() {
        return await this.aiServiceClient.listDatasets();
    }

    async processRacingSessionForAI(sessionId: string, sessionData: any[], metadata: any) {
        // Transform racing session data for AI analysis
        const aiDataset = {
            session_id: sessionId,
            session_name: metadata.sessionName || 'Racing Session',
            session_data: sessionData,
            metadata: metadata
        };

        return await this.analyzeRacingSession(aiDataset);
    }

    async askQuestionAboutSession(sessionId: string, question: string, userId?: string) {
        const query: QueryRequest = {
            question: question,
            dataset_id: sessionId,
            user_id: userId,
            context: {
                type: 'racing_session',
                session_id: sessionId
            }
        };

        return await this.processQuery(query);
    }

    async getSessionInsights(sessionId: string) {
        const analysis: AnalysisRequest = {
            dataset_id: sessionId,
            analysis_type: 'performance',
            parameters: {
                include_trends: true,
                include_recommendations: true
            }
        };

        return await this.analyzeDataset(analysis);
    }

    async detectRacingPatterns(sessionId: string) {
        return await this.aiServiceClient.detectRacingPatterns(sessionId);
    }

    async getPerformanceScore(sessionId: string) {
        return await this.aiServiceClient.getPerformanceScore(sessionId);
    }

    async getSectorAnalysis(sessionId: string) {
        return await this.aiServiceClient.getSectorAnalysis(sessionId);
    }

    async predictOptimalLapTime(sessionId: string) {
        return await this.aiServiceClient.predictOptimalLapTime(sessionId);
    }

    async healthCheck() {
        return await this.aiServiceClient.checkHealth();
    }
}
