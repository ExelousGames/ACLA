import { Strategy } from 'passport-local';
import { PassportStrategy } from '@nestjs/passport';
import { Injectable, UnauthorizedException } from '@nestjs/common';
import { AuthService } from '../shared/auth/auth.service';

//user name and password based authentication. used when user first time login
@Injectable()
export class LocalStrategy extends PassportStrategy(Strategy) {
  constructor(private authService: AuthService) {
    //req.body must have a matching 'email' and 'password'
    super({
      usernameField: 'email',
      passwordField: 'password'
    });
  }

  // Passport expects a validate() method with the following signature: validate(username: string, password:string): any
  /** The validate() method for any Passport strategy will follow a similar pattern, varying only in the details of how credentials are represented. 
   * If a user is found and the credentials are valid, the user is returned so Passport can complete its tasks
   *  (e.g., creating the user property on the Request object), and the request handling pipeline can continue. If it's not found, 
   * we throw an exception and let our exceptions layer handle it. */
  async validate(email: string, password: string): Promise<any> {
    const user = await this.authService.validateUser(email, password);
    if (!user) {
      console.log();
      throw new UnauthorizedException();
    }
    return user;
  }
}