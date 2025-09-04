import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UserInfoModule } from './modules/user-info/user-info.module';
import { MongooseModule } from '@nestjs/mongoose';
import { ThrottlerModule } from '@nestjs/throttler';
import { AuthModule } from './shared/auth/auth.module';
import { AuthorizationModule } from './shared/authorization/authorization.module';
import { RacingMapModule } from './modules/map/racing-map.module';
import { RacingSessionModule } from './modules/racing-session/racing-session.module';
import { AiModelModule as UserSessionAiModelModule } from './modules/user-session-ai-model/user-session-ai-model.module';
import { AiModelModule } from './modules/ai-model/ai-model.module';
import { ChunkModule } from './shared/chunk-service/chunk.module';

@Module({

  //MongooseModule.forRoot --- this is the port where the database is connected
  imports: [
    ThrottlerModule.forRoot([{ limit: 10, ttl: 60 }]),
    //getting Environment variable from .env coming from docker-compose.yaml
    // address is 'mongodb_c' since we are connecting another docker. 'mongodb_c' is the name of the db docker
    MongooseModule.forRoot('mongodb://' + process.env.MONGO_CLIENTNAME + ':' + process.env.MONGO_CLIENTPASSWORD + '@' + process.env.MONGO_URL + ':27017/ACLA'),
    ChunkModule,
    AuthModule,
    AuthorizationModule,
    RacingMapModule,
    UserInfoModule,
    RacingSessionModule,
    UserSessionAiModelModule,
    AiModelModule
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule { }
