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

    /**
     * Validate user credentials
     * @param email User's email
     * @param pass User's password
     * @returns User object if valid, otherwise null
     */
    async validateUser(email: string, pass: string): Promise<any> {

        const user = await this.usersService.findOneWithEmail(email);
        // Use PasswordService to compare the plain text password with the hashed password
        if (user && await this.passwordService.comparePassword(pass, user.password)) {
            // Convert Mongoose document to plain object and remove password
            const userObj = JSON.parse(JSON.stringify(user));
            const { password, ...result } = userObj;
            return result;
        }
        return null;
    }

    //after successfully login, system give user a JWT token for continued access
    async giveJWTToken(userinfo: any) {
        const payload = { username: userinfo.email, sub: userinfo.id }; // Use 'id' instead of 'userId'
        return {
            //generate our JWT from a subset of the user object properties,  //which we then return as a simple object with a single access_token property
            access_token: this.jwtService.sign(payload, { secret: jwtConstants.secret }),
            userId: userinfo.id
        };
    }

    // Hash password using PasswordService
    async hashPassword(password: string): Promise<string> {
        return await this.passwordService.hashPassword(password);
    }
}
