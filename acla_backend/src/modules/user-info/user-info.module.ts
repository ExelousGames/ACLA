import { Module } from '@nestjs/common';
import { UserInfoController } from './user-info.controller';
import { UserInfoService } from './user-info.service';
import { MongooseModule } from '@nestjs/mongoose';
import { UserInfo, UserInfoSchema } from '../../schemas/user-info.schema';
import { AuthService } from 'src/shared/auth/auth.service';
import { JwtService } from '@nestjs/jwt';
import { AuthorizationModule } from '../../shared/authorization/authorization.module';
import { PermissionsGuard } from '../../common/guards/permissions.guard';
import { PasswordService } from 'src/shared/utils/password.service';

@Module({

  //The MongooseModule provides the forFeature() method to configure the module, including defining which models should be registered in the current scope.
  //  If you also want to use the models in another module, add MongooseModule to the exports section of CatsModule and import CatsModule in the other module.
  imports: [
    MongooseModule.forFeature([{ name: UserInfo.name, schema: UserInfoSchema }]),
    AuthorizationModule
  ],
  controllers: [UserInfoController],
  providers: [UserInfoService, AuthService, JwtService, PermissionsGuard, PasswordService],

  //(services, repositories, etc.) that the current module wants to share
  exports: [UserInfoService]
})
export class UserInfoModule { }
