import { UserInfoService } from './user-info.service';
import { userInfoDto } from './dto/user-info.model';
import { CreateUserInfoDto } from './dto/create-user.dto';
import { UserInfo } from './schemas/user-info.schema';
export declare class UserInfoController {
    private userinfoService;
    constructor(userinfoService: UserInfoService);
    getUser(id: any): userInfoDto;
    createUser(createUserInfoDto: CreateUserInfoDto): UserInfo;
    deleteUser(id: string): void;
}
