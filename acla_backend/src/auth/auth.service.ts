import { Injectable } from '@nestjs/common';
import { UserInfoService } from '../user-info/user-info.service';
import { JwtService } from '@nestjs/jwt';

@Injectable()
export class AuthService {

    constructor(
        private usersService: UserInfoService,
        private jwtService: JwtService,
    ) {}

    async validateUser(username: string, pass: string): Promise<any> {
        const user = await this.usersService.findOne(username);

        //TODO: use hash to compare incoming passowrd
        if (user && user.password === pass) {
            const { password, ...result } = user;
            return result;
        }
        return null;
    }

    //generate our JWT from a subset of the user object properties, which we then return as a simple object with a single access_token property
    async login(user: any) {
        const payload = { username: user.username, sub: user.userId };
        return {
            access_token: this.jwtService.sign(payload),
        };
    }
}
