
import { ExtractJwt, Strategy } from 'passport-jwt';
import { PassportStrategy } from '@nestjs/passport';
import { Injectable } from '@nestjs/common';
import { jwtConstants } from './constants';
import { UserInfoService } from '../modules/user-info/user-info.service';

//jwt is used for further authentication after login. user will not require user name and password for further request.
@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(private userInfoService: UserInfoService) {
    super({
      //supplies the method by which the JWT will be extracted from the Request
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),

      //delegates the responsibility of ensuring that a JWT has not expired to the Passport module
      ignoreExpiration: false,

      //TODO: Use asymmetrical encryption

      //Supplying a symmetric secret for signing the token. As cautioned earlier, do not expose this secret publicly.
      secretOrKey: jwtConstants.secret,
    });
  }

  //Passport first verifies the JWT's signature and decodes the JSON. It then invokes our validate() method passing the decoded JSON as its single parameter. 
  // Based on the way JWT signing works, we're guaranteed that we're receiving a valid token that we have previously signed and issued to a valid user.
  //As a result of all this, our response to the validate() callback is trivial: we simply return an object containing the userId and username properties. 
  async validate(payload: any) {
    // Get user with permissions for authorization
    const user = await this.userInfoService.findOneWithPermissions(payload.username);
    return {
      userId: payload.sub,
      username: payload.username,
      user: user // Include full user object with permissions
    };
  }
}
