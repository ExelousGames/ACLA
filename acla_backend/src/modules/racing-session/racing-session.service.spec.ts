import { Test, TestingModule } from '@nestjs/testing';
import { RacingSessionService } from './racing-session.service';
import { getModelToken } from '@nestjs/mongoose';
import { RacingSession } from 'src/schemas/racing-session.schema';
import { GridFSService } from '../gridfs/gridfs.service';

describe('RacingSessionService', () => {
  let service: RacingSessionService;
  let racingSessionModel: any;
  let gridfsService: any;

  beforeEach(async () => {
    racingSessionModel = {
      find: jest.fn(),
      findOne: jest.fn(),
    };
    gridfsService = {
      downloadJSONStream: jest.fn(),
      getFileSize: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        RacingSessionService,
        { provide: getModelToken(RacingSession.name), useValue: racingSessionModel },
        { provide: GridFSService, useValue: gridfsService },
      ],
    }).compile();

    service = module.get<RacingSessionService>(RacingSessionService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('lists only sessions for the requested user for analysis', async () => {
    racingSessionModel.find.mockReturnValue({
      select: jest.fn().mockReturnValue({
        exec: jest.fn().mockResolvedValue([
          {
            _id: { toString: () => 'session-1' },
            session_name: 'Race 1',
            map: 'Brands Hatch',
            car_name: 'BMW',
            user_id: 'user-1',
            totalDataPoints: 100,
            totalChunks: 2,
            chunkSize: 50,
            dataChunkFileIds: ['file-1', 'file-2'],
          },
        ]),
      }),
    });

    await expect(service.listUserSessionsForAnalysis('user-1')).resolves.toEqual([
      {
        sessionId: 'session-1',
        session_name: 'Race 1',
        map: 'Brands Hatch',
        car_name: 'BMW',
        userId: 'user-1',
        totalDataPoints: 100,
        totalChunks: 2,
        chunkSize: 50,
      },
    ]);
    expect(racingSessionModel.find).toHaveBeenCalledWith({ user_id: 'user-1' });
  });
});
