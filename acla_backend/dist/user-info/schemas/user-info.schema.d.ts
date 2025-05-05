export declare class UserInfo {
    id: string;
    username: string;
}
export declare const UserInfoSchema: import("mongoose").Schema<UserInfo, import("mongoose").Model<UserInfo, any, any, any, import("mongoose").Document<unknown, any, UserInfo, any> & UserInfo & {
    _id: import("mongoose").Types.ObjectId;
} & {
    __v: number;
}, any>, {}, {}, {}, {}, import("mongoose").DefaultSchemaOptions, UserInfo, import("mongoose").Document<unknown, {}, import("mongoose").FlatRecord<UserInfo>, {}> & import("mongoose").FlatRecord<UserInfo> & {
    _id: import("mongoose").Types.ObjectId;
} & {
    __v: number;
}>;
