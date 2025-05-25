import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UserInfoModule } from './user-info/user-info.module';
import { MongooseModule } from '@nestjs/mongoose';
import { ThrottlerModule } from '@nestjs/throttler';

@Module({

  //MongooseModule.forRoot --- this is the port where the database is connected
  imports: [UserInfoModule,
    ThrottlerModule.forRoot([{ limit: 10, ttl: 60 }]),
    //getting Environment variable from .env coming from docker-compose.yaml
    // address is 'mongodb_c' since we are connecting another docker. 'mongodb_c' is the name of the db docker
    MongooseModule.forRoot('mongodb://' + process.env.MONGO_ADMINUSERNAME + ':' + process.env.MONGO_ADMINPASSWORD + '@' + process.env.MONGO_URL + ':27017')],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule { }
