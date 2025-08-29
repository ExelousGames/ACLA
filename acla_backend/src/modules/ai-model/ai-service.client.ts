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

}
