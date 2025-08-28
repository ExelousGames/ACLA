export class SessionBasicInfoListDto {
    list: {
        name: string,
        id: string
    }[] = []
}

export class UploadReacingSessionInitDto {
    sessionName: string;
    mapName: string;
    userEmail: string;
}

export class UploadReacingSessionProgressDto {
    sessionName: string;
    maoName: string;
    userEmail;
}

export class RacingSessionDetailedInfoDto {
    session_name: string;
    id: string;
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