export class AiQueryDto {
    question: string;
    dataset_id?: string;
    user_id?: string;
    context?: any;
}

export class AiAnalysisDto {
    dataset_id: string;
    analysis_type: string;
    parameters?: any;
}

export class AiDatasetUploadDto {
    id?: string;
    name: string;
    data: any[];
}

export class AiResponseDto {
    answer?: string;
    data?: any;
    visualization?: any;
    suggested_actions?: string[];
}

export class RacingSessionAiRequestDto {
    session_id: string;
    question?: string;
}

export class RacingSessionInsightsDto {
    session_id: string;
    analysis_type?: string;
    include_trends?: boolean;
    include_recommendations?: boolean;
}
