export class SessionBasicInfoListDto {
    list: {
        name: string,
        sessionId: string
    }[] = []
}

export class UploadReacingSessionInitDto {
    sessionName: string;
    mapName: string;
    carName: string;
    userId: string;
}

export class UploadReacingSessionProgressDto {
    sessionName: string;
    mapName: string;
    carName: string;
    userId: string;
}

export class RacingSessionDetailedInfoDto {
    session_name: string;
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
    totalChunks: number;
    sessionMetadata: {
        sessionId: string;
        session_name: string;
        map: string;
        car_name: string;
        userId: string;
        dataSize: number;
        chunkCount: number;
    }[];
}

export class SessionChunkDto {
    downloadId: string;
    sessionId: string;
    chunkIndex: number;
    totalChunks: number;
    data: any[];
    isComplete: boolean;
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
    guidance_type: string = "both"; // "actions", "behavior", or "both"
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