export class CreateAiModelDto {
    modelType: string;
    modelData: any;
    metadata?: any;
    isActive?: boolean;
}

export class UpdateAiModelDto {
    modelType?: string;
    modelData?: any;
    metadata?: any;
    isActive?: boolean;
}


export class TelemetryDataDto {
    telemetryData: Record<string, any>;
}