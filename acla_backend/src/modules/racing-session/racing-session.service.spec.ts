import { Test, TestingModule } from '@nestjs/testing';
import { RacingSessionService } from './racing-session.service';

describe('RacingSessionService', () => {
  let service: RacingSessionService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [RacingSessionService],
    }).compile();

    service = module.get<RacingSessionService>(RacingSessionService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
