import { Body, Controller, Delete, Get,Request, Param, Post, UseGuards } from '@nestjs/common';
import { UserInfoService } from './user-info.service';
import { LoginInfoDto } from './dto/login-info.dto';
import { CreateUserInfoDto } from './dto/create-user.dto';
import { UserInfo } from './schemas/user-info.schema';
import { AuthGuard } from '@nestjs/passport';
import { AuthService } from 'src/auth/auth.service';

@Controller('userinfo')
export class UserInfoController {

    constructor(private userinfoService: UserInfoService,private authService: AuthService) { }
    //We are using LocalStrategy localed in auth folder
    //Passport automatically creates a user object, based on the value we return from the validate() method, and assigns it to the Request object as req.user
    @UseGuards(AuthGuard('local'))
    @Post('auth/login')
    async login(@Request() req) {
        // return a JWT token for a later access
        return this.authService.login(req.user);
    }

    @UseGuards(AuthGuard('local'))
    @Post('auth/logout')
    async logout(@Request() req) {
        return req.logout();
    }

    @Post()
    createUser(@Body('infoDto') createUserInfoDto: CreateUserInfoDto): UserInfo {


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

    @UseGuards(AuthGuard('jwt'))
    @Get('profile')
    getProfile(@Request() req) {
        return req.user;
    }
}
