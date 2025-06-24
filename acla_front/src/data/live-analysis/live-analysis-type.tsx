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
    id: string;
    total_time?: number;
}

export type AllMapsBasicInfoListDto = {
    list: [{ name: string; }];
}



export type MapInfo = {

    name: string,
    mimetype: string,
    ImageData: Buffer;
    points: [{
        position: number[],
        type: number,
        index: number, //type and index are used together. some points are index sensitive
        description?: string,
        info?: string,
        variables?: [{ key: string, value: string }] //any word match {key} in description or info will be replaced with the value
    }]
}

export type UploadReacingSessionInitDto = {
    sessionName: string;
    mapName: string;
    userEmail: string;
}

export type UploadReacingSessionInitReturnDto = {
    uploadId: string;

}


export type SessionBasicInfoListDto = {
    list: {
        name: string,
        id: string
    }[]
}

export type RacingSessionDetailedInfoDto = {
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
        variables: [{ key: string, value: string }] //any word match {key} in description or info will be replaced with the value
    }[];
    data: any[];
}