import { Injectable } from '@nestjs/common';
import { userInfoDto } from './dto/user-info.model';
import { v4 as uuid } from 'uuid';
import { CreateUserInfoDto } from './dto/create-user.dto';
import { UserInfo } from './schemas/user-info.schema';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';

@Injectable()
export class UserInfoService {

    constructor(@InjectModel(UserInfo.name) private userInfoModel: Model<UserInfo>) { }

    getUser(id: string): userInfoDto {
        let info: userInfoDto = new userInfoDto;
        return info;
    }

    async createUser(createUserInfoDto: CreateUserInfoDto): Promise<UserInfo> {

        const newUserInfo: UserInfo = {
            id: uuid(),
            username: createUserInfoDto.name
        };

        const createdInfo = new this.userInfoModel(newUserInfo);


        return createdInfo.save();

    }

    deleteTask(id: string): void {
    }
}
