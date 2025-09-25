import { Types } from 'mongoose';

export class AiQueryDto {
    query: string;
    userId: Types.ObjectId;
    sessionId?: string;
    trackName?: string;
    contextType?: 'telemetry' | 'performance' | 'setup' | 'general';
    includeTelemetryData?: boolean;
    includeRealtimeAlerts?: boolean;
}

export class AiQueryResponseDto {
    answer: string;
    functionCalls?: any[];
    telemetryInsights?: any;
    recommendations?: string[];
    realtimeAlerts?: RealtimeAlert[];
    confidence: number;
    processingTime: number;
}

export class RealtimeAlert {
    type: 'brake' | 'throttle' | 'steering' | 'tyre_temp' | 'fuel' | 'damage';
    severity: 'low' | 'medium' | 'high' | 'critical';
    message: string;
    timestamp: Date;
    triggerData?: any;
    suggestedAction?: string;
}

export class TelemetryAnalysisDto {
    sessionId: string;
    telemetryData: any;
    analysisType: 'realtime' | 'post_session' | 'comparative';
    trackName: string;
    carModel?: string;
    enableRealtimeAlerts?: boolean;
}

export class BackendFunctionCallDto {
    functionName: string;
    parameters: any;
    userId: Types.ObjectId;
    context?: any;
}
