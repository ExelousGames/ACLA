import { Test, TestingModule } from '@nestjs/testing';
import { RacingMapController } from './racing-map.controller';

describe('MapController', () => {
  let controller: RacingMapController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [RacingMapController],
    }).compile();

    controller = module.get<RacingMapController>(RacingMapController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
