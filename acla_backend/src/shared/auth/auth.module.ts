import { Module } from '@nestjs/common';
import { AuthService } from './auth.service';
import { UserInfoModule } from '../../modules/user-info/user-info.module';
import { PassportModule } from '@nestjs/passport';
import { LocalStrategy } from '../../common/local.strategy';
import { JwtModule } from '@nestjs/jwt';
import { jwtConstants } from '../../common/constants';
import { JwtStrategy } from '../../common/jwt.strategy';
import { PasswordService } from '../utils/password.service';

@Module({
  imports: [
    UserInfoModule,
    PassportModule,
    JwtModule.register({
      global: true,
      secret: jwtConstants.secret,
      signOptions: { expiresIn: '60s' },
    }),
  ],
  providers: [AuthService, LocalStrategy, JwtStrategy, PasswordService],
  exports: [AuthService],
})
export class AuthModule { }
