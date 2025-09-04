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

    trackName: string;
    carName: string;
    modelType: string; // lap_time_prediction, sector_time_optimization, etc.
    modelData: any;
    metadata: any;
    isActive: boolean; // Whether this model version is active
}


export class TelemetryDataDto {
    trackName: string;
    carName: string;
    telemetryData: Record<string, any>;
}