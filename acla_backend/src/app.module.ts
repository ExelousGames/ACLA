import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UserInfoModule } from './modules/user-info/user-info.module';
import { MongooseModule } from '@nestjs/mongoose';
import { ThrottlerModule } from '@nestjs/throttler';
import { AuthModule } from './shared/auth/auth.module';
import { RacingMapModule } from './modules/map/racing-map.module';

@Module({

  //MongooseModule.forRoot --- this is the port where the database is connected
  imports: [
    ThrottlerModule.forRoot([{ limit: 10, ttl: 60 }]),
    //getting Environment variable from .env coming from docker-compose.yaml
    // address is 'mongodb_c' since we are connecting another docker. 'mongodb_c' is the name of the db docker
    MongooseModule.forRoot('mongodb://' + process.env.MONGO_CLIENTNAME + ':' + process.env.MONGO_CLIENTPASSWORD + '@' + process.env.MONGO_URL + ':27017/ACLA'),
    AuthModule,
    RacingMapModule,
    UserInfoModule,],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule { }
