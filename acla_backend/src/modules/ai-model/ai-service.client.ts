import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import axios from 'axios';

export interface QueryRequest {
    question: string;
    dataset_id?: string;
    user_id?: string;
    context?: any;
}


export interface TrainModelRequest {
    session_id: string;
    telemetry_data: any[];
    target_variable?: string;
    model_type?: string;
    user_id?: string;
    existing_model_data?: string;
    session_metadata?: Record<string, any>;
}

export interface TrainModelResponse {
    message: string;
    trained_model: any;
    instructions: string;
}

@Injectable()
export class AiServiceClient {
    private readonly aiServiceUrl: string;

    constructor() {
        this.aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000';
    }

    //ask ai service to train ai model, and return the trained model back
    async trainModel(request: TrainModelRequest): Promise<TrainModelResponse> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/train-model`, request);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service model training failed: ${error.message}`,
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
