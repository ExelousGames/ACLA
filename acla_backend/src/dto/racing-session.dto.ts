import { GameRecordedFrom } from 'src/racing-session-game';

export class SessionBasicInfoListDto {
    list: {
        name: string,
        sessionId: string
    }[] = []
}

export class MapBasicInfoListDto {
    list: {
        name: string
    }[] = []
}

export class UploadReacingSessionInitDto {
    sessionName: string;
    mapName: string;
    carName: string;
    userId: string;
    game_recorded_from: GameRecordedFrom;
}

export class UploadReacingSessionProgressDto {
    sessionName: string;
    mapName: string;
    carName: string;
    userId: string;
}

export class RacingSessionDetailedInfoDto {
    session_name: string;
    game_recorded_from: GameRecordedFrom;
    userId: string;
    map: string;
    user_email: string;

    points: {
        id: number,
        position_x: number,
        position_y: number,
        description: string,
        info: string,
        variables: { key: string, value: string }[] //any word match {key} in description or info will be replaced with the value
    }[];

    //recorded telemetry data
    data: any[];
}

export class AllSessionsInitResponseDto {
    downloadId: string;
    totalSessions: number;
    totalChunks?: number; // Legacy field for backward compatibility
    sessionMetadata: {
        sessionId: string;
        session_name: string;
        game_recorded_from: GameRecordedFrom;
        map: string;
        car_name: string;
        userId: string;
        dataSize: number;
        chunkCount?: number; // Legacy field
        fileSize?: number; // New field for streaming
        dataPoints?: number; // New field for streaming
    }[];
}

export class AnalysisSessionMetadataDto {
    sessionId: string;
    session_name: string;
    game_recorded_from: GameRecordedFrom;
    map: string;
    car_name: string;
    userId: string;
    totalDataPoints: number;
    totalChunks: number;
    chunkSize: number;
}

export class UserSessionsAnalysisInitResponseDto {
    userId: string;
    totalSessions: number;
    sessions: AnalysisSessionMetadataDto[];
}

export class SessionChunkDto {
    downloadId: string;
    sessionId: string;
    chunkIndex?: number; // Legacy field for backward compatibility
    totalChunks?: number; // Legacy field for backward compatibility
    data?: any[]; // Legacy field for backward compatibility
    isComplete?: boolean; // Legacy field for backward compatibility
    // New streaming fields
    filePath?: string;
    fileSize?: number;
    contentType?: string;
    dataPoints?: number;
}

export class AllSessionsChunkRequestDto {
    downloadId: string;
    sessionId: string;
    trackName: string;
    carName: string;
    chunkIndex: number;

}

export class ImitationLearningGuidanceRequestDto {
    current_telemetry: { [key: string]: any };
    track_name: string;
    car_name: string;
    user_id?: string;
}

export class ImitationLearningGuidanceResponseDto {
    message: string;
    guidance_result: any;
    timestamp?: string;
    recommendations?: { [key: string]: any };
    confidence_score?: number;
    success: boolean;
}

export class OpportunityForecastRequestDto {
    telemetry_data: { [key: string]: any }[];
    horizon_seconds?: number;
    top_k?: number;
}

export class OpportunityForecastOpportunityDto {
    label_id: string;
    label_name: string;
    parent_label: string;
    probability: number;
    circuit_section_id?: string;
    circuit_section_name?: string;
}

export class OpportunityForecastResponseDto {
    status: string;
    model_status?: string;
    horizon_seconds: number;
    opportunities: OpportunityForecastOpportunityDto[];
    circuit_section_match?: any;
}

export class TrackCornerKnowledgeRequestDto {
    track_name: string;
    corner_name: string;
    normalized_position?: number;
    trigger_position?: number;
    current_telemetry?: { [key: string]: any };
}

export class TrackCornerKnowledgeResponseDto {
    status: string;
    message?: string;
    reason?: string;
    track_knowledge: any;
    normalized_position?: number;
    trigger_position?: number;
}

export class SegmentClassificationRequestDto {
    session_id: string;
}

export class LiveBaselineTimeGapDto {
    start_ms: number;
    end_ms: number;
    delta_ms: number;
}

export class ExpertReferenceRowDto {
    raw_index: number;
    expert_time_difference: number;
    expert_optimal_time: number;
    expert_optimal_player_pos_x: number;
    expert_optimal_player_pos_y: number;
    expert_optimal_player_pos_z: number;
    Graphics_normalized_car_position: number;
    expert_optimal_throttle: number;
    expert_optimal_brake: number;
    expert_optimal_gear: number;
}

export class SegmentClassificationSegmentDto {
    id?: string;
    labels: string[];
    track_section?: string;
    start_index: number;
    end_index: number;
    time_gap?: LiveBaselineTimeGapDto;
    expert_reference_data: ExpertReferenceRowDto[];
}

export class SegmentClassificationResponseDto {
    status: string;
    session_id: string;
    samples_analyzed: number;
    parent_segment_count: number;
    segments: SegmentClassificationSegmentDto[];
}

export class LiveBaselineAnalysisRequestDto {
    track?: string;
    car?: string;
    baseline_lap?: number;
    records: { [key: string]: any }[];
}

export class LiveBaselineAnalysisResponseDto extends SegmentClassificationResponseDto {
    expert_time_available: boolean;
}
