export class CreateAiModelDto {
    trackName?: string;
    carName?: string;
    modelType: string;
    modelData: any;
    metadata?: any;
    isActive?: boolean;
}

export class UpdateAiModelDto {
    trackName?: string;
    carName?: string;
    modelType?: string;
    modelData?: any;
    metadata?: any;
    isActive?: boolean;
}


export class TelemetryDataDto {
    trackName: string;
    carName: string;
    telemetryData: Record<string, any>;
}