import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UserInfoModule } from './user-info/user-info.module';
import { MongooseModule } from '@nestjs/mongoose';

@Module({

  //MongooseModule.forRoot --- this is the port where the database is connected
  imports: [UserInfoModule, MongooseModule.forRoot('mongodb://127.0.0.1/DBCollections')],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule { }
