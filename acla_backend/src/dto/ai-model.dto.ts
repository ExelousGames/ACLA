import { Types } from 'mongoose';

export class CreateAiModelDto {
    userId: Types.ObjectId;
    trackName: string;
    modelData: string; // Base64 encoded serialized model
    modelType: string; // lap_time_prediction, sector_time_optimization, etc.
    algorithmUsed: string; // random_forest, gradient_boosting, neural_network, etc.
    algorithmType: string; // regression, classification
    targetVariable: string; // lap_time, sector_time, etc.
    trainingMetrics: Record<string, any>; // Model performance metrics
    featureNames: string[]; // List of feature names used
    featureCount: number; // Number of features
    trainingSamples: number; // Number of training samples
    sessionMetadata?: Record<string, any>; // Additional session information
    modelVersion?: number; // Version number for incremental training
    telemetrySummary?: Record<string, any>; // Summary of telemetry data used
    recommendations?: string[]; // Training recommendations
    algorithmDescription?: string; // Description of the algorithm used
    supportsIncremental?: boolean; // Whether model supports incremental learning
    featureImportance?: Record<string, any>; // Feature importance scores
    alternativeAlgorithms?: string[]; // Alternative algorithms for this model type
    trainedAt: Date; // When the model was trained
    isActive?: boolean; // Whether this model version is active
}

export class UpdateAiModelDto {
    modelData?: string; // Base64 encoded serialized model
    modelType?: string; // lap_time_prediction, sector_time_optimization, etc.
    algorithmUsed?: string; // random_forest, gradient_boosting, neural_network, etc.
    algorithmType?: string; // regression, classification
    targetVariable?: string; // lap_time, sector_time, etc.
    trainingMetrics?: Record<string, any>; // Model performance metrics
    featureNames?: string[]; // List of feature names used
    featureCount?: number; // Number of features
    trainingSamples?: number; // Number of training samples
    modelVersion?: number; // Version number for incremental training
    telemetrySummary?: Record<string, any>; // Summary of telemetry data used
    recommendations?: string[]; // Training recommendations
    algorithmDescription?: string; // Description of the algorithm used
    supportsIncremental?: boolean; // Whether model supports incremental learning
    featureImportance?: Record<string, any>; // Feature importance scores
    alternativeAlgorithms?: string[]; // Alternative algorithms for this model type
    trainedAt?: Date; // When the model was trained
    isActive?: boolean; // Whether this model version is active
}

export class GetAiModelDto {
    userId: Types.ObjectId;
    trackName?: string;
    modelType?: string;
    algorithmUsed?: string;
    algorithmType?: string;
    activeOnly?: boolean;
    supportsIncremental?: boolean;
}

export class IncrementalTrainingDto {
    modelId: string;
    newSessionIds: string[];
    trainingParameters?: any;
    validateModel?: boolean;
}

export class ModelPredictionDto {
    modelId: string;
    inputData: any;
    predictionOptions?: any;
}

export class AiModelResponseDto {
    id: string;
    userId: Types.ObjectId;
    trackName: string;
    modelData: string; // Base64 encoded serialized model
    modelType: string; // lap_time_prediction, sector_time_optimization, etc.
    algorithmUsed: string; // random_forest, gradient_boosting, neural_network, etc.
    algorithmType: string; // regression, classification
    targetVariable: string; // lap_time, sector_time, etc.
    trainingMetrics: Record<string, any>; // Model performance metrics
    featureNames: string[]; // List of feature names used
    featureCount: number; // Number of features
    trainingSamples: number; // Number of training samples
    sessionMetadata: Record<string, any>; // Additional session information
    modelVersion: number; // Version number for incremental training
    telemetrySummary: Record<string, any>; // Summary of telemetry data used
    recommendations: string[]; // Training recommendations
    algorithmDescription: string; // Description of the algorithm used
    supportsIncremental: boolean; // Whether model supports incremental learning
    featureImportance: Record<string, any>; // Feature importance scores
    alternativeAlgorithms: string[]; // Alternative algorithms for this model type
    trainedAt: Date; // When the model was trained
    isActive: boolean; // Whether this model version is active
}
