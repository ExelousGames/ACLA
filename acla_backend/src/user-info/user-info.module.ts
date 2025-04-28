import { Module } from '@nestjs/common';
import { UserInfoController } from './user-info.controller';
import { UserInfoService } from './user-info.service';
import { MongooseModule } from '@nestjs/mongoose';
import { UserInfo, UserInfoSchema } from './schemas/user-info.schema';

@Module({
  imports: [MongooseModule.forFeature([{ name: UserInfo.name, schema: UserInfoSchema }])],
  controllers: [UserInfoController],
  providers: [UserInfoService]
})
export class UserInfoModule { }
