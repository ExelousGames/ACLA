export class CreateAiModelDto {
    trackName: string;
    carName: string;
    modelType: string;
    targetVariable: string[];
    modelData: string;
    trainingMetrics?: Record<string, any>;
    samplesProcessed?: number;
    featureNames: string[];
    modelVersion?: string;
    recommendations?: string[];
    dataQualityScore?: number;
    timestamp?: string;
    isActive?: boolean;
}

export class UpdateAiModelDto {
    trackName?: string;
    carName?: string;
    modelType?: string;
    targetVariable?: string[];
    modelData?: string;
    trainingMetrics?: Record<string, any>;
    samplesProcessed?: number;
    featureNames?: string[];
    modelVersion?: string;
    recommendations?: string[];
    dataQualityScore?: number;
    timestamp?: string;
    isActive?: boolean;
}
