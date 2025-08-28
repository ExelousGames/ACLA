import { Module } from '@nestjs/common';
import { RacingSessionService } from './racing-session.service';
import { RacingSessionController } from './racing-session.controller';
import { RacingSession, RacingSessionSchema } from 'src/schemas/racing-session.schema';
import { MongooseModule } from '@nestjs/mongoose';
import { AiServiceModule } from '../ai-service/ai-service.module';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: RacingSession.name, schema: RacingSessionSchema }]),
    AiServiceModule,
  ],
  providers: [RacingSessionService],
  controllers: [RacingSessionController],
  exports: [RacingSessionService]
})
export class RacingSessionModule { }
