import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import axios from 'axios';
import { UserACCTrackAIModel } from 'src/schemas/session-ai-model.schema';

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
    user_id?: string;
}

export interface ImitationLearningGuidanceResponse {
    message: string;
    guidance_result: any;
    timestamp?: string;
    recommendations?: { [key: string]: any };
    confidence_score?: number;
}

export interface OpportunityForecastRequest {
    telemetry_data: { [key: string]: any }[];
    horizon_seconds?: number;
    top_k?: number;
}

export interface OpportunityForecastOpportunity {
    label_id: string;
    label_name: string;
    parent_label: string;
    probability: number;
    circuit_section_id?: string;
    circuit_section_name?: string;
}

export interface OpportunityForecastResponse {
    status: string;
    model_status?: string;
    horizon_seconds: number;
    opportunities: OpportunityForecastOpportunity[];
    circuit_section_match?: any;
}

export interface TrackCornerKnowledgeRequest {
    track_name: string;
    corner_name: string;
    normalized_position?: number;
    trigger_position?: number;
    current_telemetry?: { [key: string]: any };
}

export interface TrackCornerKnowledgeResponse {
    status: string;
    track_knowledge: any;
    normalized_position?: number;
    trigger_position?: number;
}

export interface SegmentClassificationRequest {
    session_id?: string;
    telemetry_data: { [key: string]: any }[];
    track_name?: string;
    car_name?: string;
}

export interface SegmentClassificationLabel {
    label_id: string;
    label_name: string;
}

export interface SegmentClassificationSubSegment {
    start_index: number;
    end_index: number;
    labels: SegmentClassificationLabel[];
}

export interface SegmentClassificationSegment {
    id?: string;
    labels: string[];
    main_label_id: string;
    main_label_name: string;
    start_index: number;
    end_index: number;
    sub_labels: SegmentClassificationLabel[];
    sub_segments: SegmentClassificationSubSegment[];
}

export interface SegmentClassificationResponse {
    status: string;
    session_id: string;
    samples_analyzed: number;
    segment_count: number;
    segments: SegmentClassificationSegment[];
}

// Phase 2 — text-to-speech via Kokoro
export interface AnalyzeUserSessionsRequest {
    user_id: string;
}

export interface AnalyzeUserSessionsResponse {
    status: string;
    sessionAnalysis: Record<string, any>;
}

export interface VoiceSynthesizeRequest {
    text: string;
    voice?: string;       // e.g. "af_bella"; defaults to AI service's kokoro_default_voice
    speed?: number;       // 0.5..2.0
    language?: string;    // e.g. "en-us"
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

    async getOpportunityForecast(request: OpportunityForecastRequest): Promise<OpportunityForecastResponse> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/opportunity-forecast`, request);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service opportunity forecast failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async getTrackCornerKnowledge(request: TrackCornerKnowledgeRequest): Promise<TrackCornerKnowledgeResponse> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/track-corner-knowledge`, request);
            return response.data;
        } catch (error) {
            const axiosError = error as any;
            const detail = axiosError?.response?.data?.detail
                || axiosError?.response?.data?.message
                || axiosError?.message;
            const status = axiosError?.response?.status || HttpStatus.SERVICE_UNAVAILABLE;
            throw new HttpException(
                `AI Service track corner knowledge failed: ${detail}`,
                status
            );
        }
    }

    async classifySegments(request: SegmentClassificationRequest): Promise<SegmentClassificationResponse> {
        try {
            const response = await axios.post(`${this.aiServiceUrl}/racing-session/segment-classification`, request);
            return response.data;
        } catch (error) {
            const axiosError = error as any;
            const detail = axiosError?.response?.data?.detail
                || axiosError?.response?.data?.message
                || axiosError?.message;
            const status = axiosError?.response?.status || HttpStatus.SERVICE_UNAVAILABLE;
            throw new HttpException(
                `AI Service segment classification failed: ${detail}`,
                status
            );
        }
    }

    /**
     * Phase 2 — Neural text-to-speech via Kokoro.
     * Returns raw WAV bytes (audio/wav). The controller forwards these
     * to the browser, which plays them via HTMLAudioElement, replacing
     * the robotic window.speechSynthesis output.
     */
    async analyzeUserSessions(request: AnalyzeUserSessionsRequest): Promise<AnalyzeUserSessionsResponse> {
        try {
            const response = await axios.post(
                `${this.aiServiceUrl}/racing-session/analyze-user-sessions`,
                request,
                { timeout: 24 * 60 * 60 * 1000 },
            );
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service user session analysis failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    async synthesizeVoice(request: VoiceSynthesizeRequest): Promise<Buffer> {
        try {
            const response = await axios.post(
                `${this.aiServiceUrl}/voice/synthesize`,
                request,
                { responseType: 'arraybuffer' },
            );
            return Buffer.from(response.data);
        } catch (error) {
            throw new HttpException(
                `AI Service voice synthesis failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

    /** Phase 2 — list available Kokoro voices. */
    async listVoices(): Promise<{ voices: string[]; count: number }> {
        try {
            const response = await axios.get(`${this.aiServiceUrl}/voice/voices`);
            return response.data;
        } catch (error) {
            throw new HttpException(
                `AI Service voice listing failed: ${error.message}`,
                HttpStatus.SERVICE_UNAVAILABLE
            );
        }
    }

}
