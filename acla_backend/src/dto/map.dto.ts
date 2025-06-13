export class AllMapsBasicInfoListDto {
    list: MapBasicInfo[]
    constructor() {
        this.list = [];
    }
}

export type MapBasicInfo = {
    name: string
}