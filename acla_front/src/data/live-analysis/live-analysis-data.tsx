export type MapOption = {
    //key is required for child component of MapList component
    key: number,
    name?: string;
    session_count?: number;
}

export type SessionOption = {
    //key is required for child component of MapList component
    key: number,
    name: string;
    total_time: number;
}

export class AllMapsBasicInfoListDto {
    list?: [{ name: string }]
}