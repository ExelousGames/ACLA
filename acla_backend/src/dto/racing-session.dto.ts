export class SessionBasicInfoListDto {
    list: [{
        name: string,
        map: string,
    }]
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