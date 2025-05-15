import { Body, Controller, Delete, Get, Param, Post } from '@nestjs/common';
import { UserInfoService } from './user-info.service';
import { userInfoDto } from './dto/user-info.model';
import { CreateUserInfoDto } from './dto/create-user.dto';
import { UserInfo } from './schemas/user-info.schema';

@Controller('userinfo')
export class UserInfoController {

    constructor(private userinfoService: UserInfoService) { }

    @Get(':id')
    getUser(@Param('id') id): userInfoDto {
        return this.userinfoService.getUser(id);
    }

    @Post()
    createUser(@Body('infoDto') createUserInfoDto: CreateUserInfoDto): UserInfo {

        console.log("create User");
        this.userinfoService.createUser(createUserInfoDto).then(
            (dto) => {
                return dto;
            }).catch((error) => {
                console.log(error);
            });
        return new UserInfo;
    }

    @Delete(':id')
    deleteUser(@Param('id') id: string): void {
        this.userinfoService.deleteTask(id);
    }
}
