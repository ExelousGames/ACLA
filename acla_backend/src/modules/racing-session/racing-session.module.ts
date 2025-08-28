import { Module } from '@nestjs/common';
import { RacingSessionService } from './racing-session.service';
import { RacingSessionController } from './racing-session.controller';
import { RacingSession, RacingSessionSchema } from 'src/schemas/racing-session.schema';
import { MongooseModule } from '@nestjs/mongoose';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: RacingSession.name, schema: RacingSessionSchema }]),
  ],
  providers: [RacingSessionService],
  controllers: [RacingSessionController],
  exports: [RacingSessionService]
})
export class RacingSessionModule { }
