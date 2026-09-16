export type MapOption = {
    //key is required for child component of MapList component
    dataKey: number,
    name: string;
    session_count?: number;
}

export type SessionOption = {
    //key is required for child component of MapList component
    dataKey: number,
    name: string;
    SessionId: string;
    total_time?: number;
}

export type AllMapsBasicInfoListDto = {
    list: { name: string; }[];
}

export type RecordedSessionSource = DesktopGame | 'iracing_live' | 'iracing_recorded';

export type UploadReacingSessionInitDto = {
    sessionName: string;
    mapName: string;
    carName: string;
    userId: string;
    game_recorded_from: RecordedSessionSource;
}

export type UploadRacingSessionInitReturnDto = {
    uploadId: string;
}


export type SessionBasicInfoListDto = {
    list: {
        name: string,
        sessionId: string
    }[]
}

export type RacingSessionDetailedInfoDto = {
    storage?: 'cloud' | 'local';
    session_name: string;
    game_recorded_from?: RecordedSessionSource;
    SessionId: string;
    map: string;
    car: string;
    user_id: string;
    points: {
        id: number,
        position_x: number,
        position_y: number,
        description: string,
        info: string,
        variables: [{ key: string, value: string }] //any word match {key} in description or info will be replaced with the value
    }[];
    data: any[];
}
import type { DesktopGame } from 'contexts/DesktopGameContext';
