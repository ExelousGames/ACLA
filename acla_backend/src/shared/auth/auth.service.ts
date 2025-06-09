import { Injectable } from '@nestjs/common';
import { UserInfoService } from '../../modules/user-info/user-info.service';
import { JwtService } from '@nestjs/jwt';
import { jwtConstants } from '../../common/constants';

@Injectable()
export class AuthService {

    constructor(
        private usersService: UserInfoService,
        private jwtService: JwtService,
    ) { }

    async validateUser(username: string, pass: string): Promise<any> {
        const user = await this.usersService.findOne(username);
        //TODO: use hash to compare incoming password
        if (user && user.password === pass) {
            const { password, ...result } = user;
            return result;
        }
        return null;
    }

    //used by jwt authentication
    async giveJWTToken(user: any) {
        const payload = { username: user.username, sub: user.userId };
        return {
            //generate our JWT from a subset of the user object properties,  //which we then return as a simple object with a single access_token property
            access_token: this.jwtService.sign(payload, { secret: jwtConstants.secret }),
        };
    }
}
