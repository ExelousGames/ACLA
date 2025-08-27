import { Injectable } from '@nestjs/common';
import { UserInfoService } from '../../modules/user-info/user-info.service';
import { JwtService } from '@nestjs/jwt';
import { jwtConstants } from '../../common/constants';
import { PasswordService } from '../utils/password.service';

@Injectable()
export class AuthService {

    constructor(
        private usersService: UserInfoService,
        private jwtService: JwtService,
        private passwordService: PasswordService,
    ) {
    }

    async validateUser(email: string, pass: string): Promise<any> {
        const user = await this.usersService.findOne(email);
        console.log(user);
        
        // Use PasswordService to compare the plain text password with the hashed password
        if (user && await this.passwordService.comparePassword(pass, user.password)) {
            const { email, ...result } = user;
            return result;
        }
        return null;
    }

    //used by jwt authentication
    async giveJWTToken(userinfo: any) {
        const payload = { username: userinfo.email, sub: userinfo.userId };
        return {
            //generate our JWT from a subset of the user object properties,  //which we then return as a simple object with a single access_token property
            access_token: this.jwtService.sign(payload, { secret: jwtConstants.secret }),
        };
    }

    // Hash password using PasswordService
    async hashPassword(password: string): Promise<string> {
        return await this.passwordService.hashPassword(password);
    }
}
