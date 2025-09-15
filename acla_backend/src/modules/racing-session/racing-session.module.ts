import { Module, forwardRef } from '@nestjs/common';
import { RacingSessionService } from './racing-session.service';
import { RacingSessionController } from './racing-session.controller';
import { RacingSession, RacingSessionSchema } from 'src/schemas/racing-session.schema';
import { MongooseModule } from '@nestjs/mongoose';
import { AiModelModule } from '../user-session-ai-model/user-session-ai-model.module';
import { GridFSModule } from '../gridfs/gridfs.module';
import { UserInfoModule } from '../user-info/user-info.module';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: RacingSession.name, schema: RacingSessionSchema }]),
  forwardRef(() => AiModelModule),
  GridFSModule,
    UserInfoModule,
  ],
  providers: [RacingSessionService],
  controllers: [RacingSessionController],
  exports: [RacingSessionService]
})
export class RacingSessionModule { }
