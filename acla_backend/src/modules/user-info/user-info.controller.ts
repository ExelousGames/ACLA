import { Body, Controller, Delete, Get, Request, Param, Post, UseGuards } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AuthService } from 'src/shared/auth/auth.service';
import { UserInfo } from '../../schemas/user-info.schema';
import { UserInfoService } from './user-info.service';
import { CreateUserInfoDto } from 'src/dto/user.dto';

@Controller('userinfo')
export class UserInfoController {

    constructor(private userinfoService: UserInfoService, private authService: AuthService) { }

    //We are using LocalStrategy located in auth folder. our passport local strategy has a default name of 'local',
    //we refers the name in 'AuthGuard'
    //Passport automatically creates a user object (populated by passport during the passport-local authentication flow), 
    // based on the value we return from the validate() method, 
    // and assigns it to the Request object as req.user
    @UseGuards(AuthGuard('local'))
    @Post('auth/login')
    async login(@Request() req) {
        // return a JWT token for a later access
        return this.authService.giveJWTToken(req.email);
    }

    @UseGuards(AuthGuard('local'))
    @Post('auth/logout')
    async logout(@Request() req) {
        return req.logout();
    }

    @Post()
    createUser(@Body('infoDto') createUserInfoDto: CreateUserInfoDto): CreateUserInfoDto {


        this.userinfoService.createUser(createUserInfoDto).then(
            (dto) => {
                return dto;
            }).catch((error) => {

            });
        return new CreateUserInfoDto;
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
