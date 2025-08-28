import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import axios from 'axios';

export interface QueryRequest {
    question: string;
    dataset_id?: string;
    user_id?: string;
    context?: any;
}

export interface AnalysisRequest {
    dataset_id: string;
    analysis_type: string;
    parameters?: any;
}

export interface DatasetUploadRequest {
    id?: string;
    name: string;
    data: any[];
}

@Injectable()
export class AiServiceClient {
    private readonly aiServiceUrl: string;

    constructor() {
        this.aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000';
    }

    async uploadDataset(dataset: DatasetUploadRequest): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/datasets/upload`, dataset);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service dataset upload failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async processQuery(query: QueryRequest): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/query`, query);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service query failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async analyzeDataset(analysis: AnalysisRequest): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/analyze`, analysis);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service analysis failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async analyzeRacingSession(sessionData: any): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/analyze`, sessionData);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service racing session analysis failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async listDatasets(): Promise<any> {
        try {
            const response = await axios.get(`${this.aiServiceUrl}/datasets`);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service list datasets failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async checkHealth(): Promise<any> {
        try {
            const response = await axios.get(`${this.aiServiceUrl}/health`);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service health check failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async detectRacingPatterns(sessionId: string): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/patterns`, {
                session_id: sessionId
            });
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service pattern detection failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async getPerformanceScore(sessionId: string): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/performance-score`, {
                session_id: sessionId
            });
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service performance score failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async getSectorAnalysis(sessionId: string): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/sector-analysis`, {
                session_id: sessionId
            });
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service sector analysis failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async predictOptimalLapTime(sessionId: string): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/optimal-prediction`, {
                session_id: sessionId
            });
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service optimal prediction failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }
}
