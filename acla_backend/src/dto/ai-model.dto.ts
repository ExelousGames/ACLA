import { Types } from 'mongoose';

export class CreateAiModelDto {
    userId: Types.ObjectId;
    trackName: string;
    modelName: string;
    modelVersion: string;
    modelData: any;
    modelMetadata: {
        trainingSessionsCount: number;
        lastTrainingDate: Date;
        performanceMetrics: any;
        modelType: string;
        accuracy?: number;
        mse?: number;
        features: string[];
        hyperparameters?: any;
    };
    trainingSessionIds: string[];
    isActive: boolean;
    description?: string;
    validationResults?: any;
    featureImportance?: any;
    modelSize?: number;
    trainingDuration?: number;
}

export class UpdateAiModelDto {
    modelName?: string;
    modelVersion?: string;
    modelData?: any;
    modelMetadata?: {
        trainingSessionsCount: number;
        lastTrainingDate: Date;
        performanceMetrics: any;
        modelType: string;
        accuracy?: number;
        mse?: number;
        features: string[];
        hyperparameters?: any;
    };
    trainingSessionIds?: string[];
    isActive?: boolean;
    description?: string;
    validationResults?: any;
    featureImportance?: any;
    modelSize?: number;
    trainingDuration?: number;
}

export class GetAiModelDto {
    userId: Types.ObjectId;
    trackName: string;
    modelType?: string;
    activeOnly?: boolean;
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
    modelName: string;
    modelVersion: string;
    modelMetadata: any;
    trainingSessionIds: string[];
    isActive: boolean;
    description?: string;
    validationResults?: any;
    featureImportance?: any;
    modelSize?: number;
    trainingDuration?: number;
    createdAt: Date;
    updatedAt: Date;
}
