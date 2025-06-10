import { Injectable } from '@nestjs/common';
import { v4 as uuid } from 'uuid';
import { UserInfo } from '../../schemas/user-info.schema';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { CreateUserInfoDto } from 'src/dto/user.dto';

@Injectable()
export class UserInfoService {

    //Once you've registered the schema in module, you can inject a model into the Service using the @InjectModel() decorator:
    constructor(@InjectModel(UserInfo.name) private userInfoModel: Model<UserInfo>) { }

    async findOne(email: string): Promise<UserInfo | null> {
        return this.userInfoModel.findOne({ email: email }).exec();
    }

    async createUser(createUserInfoDto: CreateUserInfoDto): Promise<CreateUserInfoDto> {

        const newUserInfo: UserInfo = {
            id: uuid(),
            password: "",
            email: createUserInfoDto.email
        };

        const createdInfo = new this.userInfoModel(newUserInfo);

        createdInfo.save();
        return new CreateUserInfoDto();

    }

    deleteTask(id: string): void {
    }
}
