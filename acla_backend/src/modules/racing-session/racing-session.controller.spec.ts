import { Test, TestingModule } from '@nestjs/testing';
import { RacingSessionController } from './racing-session.controller';

describe('RacingSessionController', () => {
  let controller: RacingSessionController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [RacingSessionController],
    }).compile();

    controller = module.get<RacingSessionController>(RacingSessionController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
