import { userInfoDto } from './dto/user-info.model';
import { CreateUserInfoDto } from './dto/create-user.dto';
import { UserInfo } from './schemas/user-info.schema';
import { Model } from 'mongoose';
export declare class UserInfoService {
    private userInfoModel;
    constructor(userInfoModel: Model<UserInfo>);
    getUser(id: string): userInfoDto;
    createUser(createUserInfoDto: CreateUserInfoDto): Promise<UserInfo>;
    deleteTask(id: string): void;
}
