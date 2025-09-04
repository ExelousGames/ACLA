import { Injectable } from '@nestjs/common';
import { v4 as uuid } from 'uuid';
import { UserInfo } from '../../schemas/user-info.schema';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import { CreateUserInfoDto, UpdateUserPermissionsDto, UpdateUserRolesDto, UpdateUserPasswordDto } from 'src/dto/user.dto';
import { PasswordService } from 'src/shared/utils/password.service';

@Injectable()
export class UserInfoService {

    //Once you've registered the schema in module, you can inject a model into the Service using the @InjectModel() decorator:
    constructor(
        @InjectModel(UserInfo.name) private userInfoModel: Model<UserInfo>,
        private passwordService: PasswordService
    ) { }

    async findOneWithEmail(email: string): Promise<UserInfo | null> {
        return this.userInfoModel.findOne({ email: email }).exec();
    }

    async findOneWithPermissions(id: string): Promise<UserInfo | null> {

        //path is the field name in UserInfo schema
        //model is the name of the model we are referencing to
        //populate is a mongoose method to populate the referenced documents
        return this.userInfoModel
            .findOne({ id: id })
            .populate({
                path: 'permissions',
                model: 'Permission'
            })
            .populate({
                path: 'roles',
                model: 'Role',
                populate: {
                    path: 'permissions',
                    model: 'Permission'
                }
            })
            .exec();
    }

    async createUser(createUserInfoDto: CreateUserInfoDto): Promise<CreateUserInfoDto> {
        // Hash the password before saving
        const hashedPassword = await this.passwordService.hashPassword(createUserInfoDto.password);

        const newUserInfo: UserInfo = {
            id: uuid(),
            password: hashedPassword,
            email: createUserInfoDto.email,
            roles: [],
            permissions: [],
            isActive: true,
            createdAt: new Date(),
            lastLogin: new Date()
        };

        const createdInfo = new this.userInfoModel(newUserInfo);

        await createdInfo.save();
        return new CreateUserInfoDto();
    }

    deleteTask(id: string): void {
    }

    async updateUserPermissions(updatePermissionsDto: UpdateUserPermissionsDto): Promise<UserInfo | null> {
        // Convert string IDs to ObjectId
        const permissionObjectIds = updatePermissionsDto.permissions.map(permissionId => new Types.ObjectId(permissionId));

        return this.userInfoModel.findOneAndUpdate(
            { id: updatePermissionsDto.userId },
            { permissions: permissionObjectIds },
            { new: true }
        ).exec();
    }

    async updateUserRoles(updateRolesDto: UpdateUserRolesDto): Promise<UserInfo | null> {
        // Convert string IDs to ObjectId
        const roleObjectIds = updateRolesDto.roles.map(roleId => new Types.ObjectId(roleId));

        return this.userInfoModel.findOneAndUpdate(
            { id: updateRolesDto.userId },
            { roles: roleObjectIds },
            { new: true }
        ).exec();
    }

    async updateUserPassword(updatePasswordDto: UpdateUserPasswordDto): Promise<UserInfo | null> {
        // Hash the new password
        const hashedPassword = await this.passwordService.hashPassword(updatePasswordDto.newPassword);

        return this.userInfoModel.findOneAndUpdate(
            { id: updatePasswordDto.userId },
            { password: hashedPassword },
            { new: true }
        ).exec();
    }
}
