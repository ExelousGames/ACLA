import { Test, TestingModule } from '@nestjs/testing';
import { RacingMapService } from './racing-map.service';

describe('MapService', () => {
  let service: RacingMapService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [RacingMapService],
    }).compile();

    service = module.get<RacingMapService>(RacingMapService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
