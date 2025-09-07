import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import axios from 'axios';
import { UserACCTrackAIModel } from 'src/schemas/session-ai-model.schema';

export interface QueryRequest {
    question: string;
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
}

export interface MultipleTrainingRequest {
    session_id: string
    telemetry_data: any[];

    /**
     * example of models_config
        {
        "config_id": "rf_model",
        "target_variable": "lap_time", 
        "model_type": "lap_time_prediction",
        "preferred_algorithm": "random_forest",
        "existing_model_data": data
        }
    */
    models_config: ModelsConfig[];  // List of model configurations to train
    user_id?: string
    parallel_training: boolean;  // Whether to train models in parallel or sequentially
}

export interface ModelsConfig {
    config_id: string;
    target_variable: string; //what do you want to predict in the telemetry data
    model_type: string; // find the model type in app/models/telemetry_models.py
    preferred_algorithm?: string;

    // Optional existing model data from database SessionAIModel schema for incremental training
    existing_model_data?: any | null;
}

export interface TrainModelResponse {
    success: boolean,
    model_data: any,
    model_type: string,
    algorithm_used: string,
    algorithm_type: string,
    target_variable: string,
    user_id: string,
    training_metrics: any,
    feature_names: string[],
    features_count: number,
    samples_processed: number,
    model_version: string,
    recommendations: string[],
    algorithm_description: string,
    algorithm_strengths: string[],
    training_time: string,
    data_quality_score: number,
    timestamp: string,
}

export interface TrainModelsResponse {
    message: string;
    session_id: string;
    total_models_requested: number;
    successful_trainings: number;
    failed_trainings: number;
    // Mapping of model IDs to their training results, contains same as TrainModelResponse
    training_results: { [key: string]: TrainModelResponse };
    instructions: string;
}

export interface ImitationLearningGuidanceRequest {
    current_telemetry: { [key: string]: any };
    track_name: string;
    car_name: string;
    guidance_type: string; // "actions", "behavior", or "both"
    user_id?: string;
}

export interface ImitationLearningGuidanceResponse {
    message: string;
    guidance_result: any;
    timestamp?: string;
    recommendations?: { [key: string]: any };
    confidence_score?: number;
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

    //ask ai service to train ai model, and return the trained model back
    async trainModels(request: MultipleTrainingRequest): Promise<TrainModelsResponse> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/train-multiple-models`, request);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `request for AI Service model training failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }


    async processQuery(query: QueryRequest): Promise<any> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/naturallanguagequery`, query);
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

    async getImitationLearningGuidance(request: ImitationLearningGuidanceRequest): Promise<ImitationLearningGuidanceResponse> {
        try {

            const response = await axios.post(`${this.aiServiceUrl}/racing-session/imitation-learning-guidance`, request);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service imitation learning guidance failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

}
