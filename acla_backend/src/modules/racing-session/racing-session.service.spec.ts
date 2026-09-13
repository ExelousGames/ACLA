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
      findById: jest.fn(),
      create: jest.fn(),
    };
    gridfsService = {
      uploadJSON: jest.fn(),
      downloadJSONStream: jest.fn(),
      getFileSize: jest.fn(),
      downloadJSON: jest.fn(),
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

  it('persists game metadata when creating a session from telemetry data', async () => {
    gridfsService.uploadJSON.mockResolvedValue('507f1f77bcf86cd799439012');
    racingSessionModel.create.mockResolvedValue({ _id: 'session-1' });

    await service.createRacingSession(
      'Race 1',
      'Monza',
      'GT3',
      'user-1',
      'iracing',
      [{ speed: 120 }],
    );

    expect(racingSessionModel.create).toHaveBeenCalledWith(expect.objectContaining({
      session_name: 'Race 1',
      map: 'Monza',
      car_name: 'GT3',
      user_id: 'user-1',
      game_recorded_from: 'iracing',
    }));
  });

  it('persists game metadata when creating a session from uploaded chunks', async () => {
    racingSessionModel.create.mockResolvedValue({ _id: 'session-1' });

    await service.createRacingSessionFromChunks(
      'Race 1',
      'Monza',
      'GT3',
      'user-1',
      'ac',
      ['507f1f77bcf86cd799439012'] as any,
      50,
      1000,
    );

    expect(racingSessionModel.create).toHaveBeenCalledWith(expect.objectContaining({
      game_recorded_from: 'ac',
      totalChunks: 1,
      totalDataPoints: 50,
    }));
  });

  it('returns game metadata in detailed session information', async () => {
    racingSessionModel.findOne.mockReturnValue({
      exec: jest.fn().mockResolvedValue({
        session_name: 'Race 1',
        game_recorded_from: 'acc',
        map: 'Monza',
        user_id: 'user-1',
        points: [],
      }),
    });

    await expect(service.retrieveSessionDetailedInfo('user-1')).resolves.toMatchObject({
      session_name: 'Race 1',
      game_recorded_from: 'acc',
      map: 'Monza',
      userId: 'user-1',
    });
  });

  it('lists only sessions for the requested user for analysis', async () => {
    const exec = jest.fn().mockResolvedValue([
      {
        _id: { toString: () => 'session-1' },
        session_name: 'Race 1',
        game_recorded_from: 'acc',
        map: 'Brands Hatch',
        car_name: 'BMW',
        user_id: 'user-1',
        totalDataPoints: 100,
        totalChunks: 2,
        chunkSize: 50,
        dataChunkFileIds: ['file-1', 'file-2'],
      },
    ]);
    const limit = jest.fn().mockReturnValue({ exec });
    const sort = jest.fn().mockReturnValue({ limit });
    const select = jest.fn().mockReturnValue({ sort });
    racingSessionModel.find.mockReturnValue({
      select,
    });

    await expect(service.listUserSessionsForAnalysis('user-1')).resolves.toEqual([
      {
        sessionId: 'session-1',
        session_name: 'Race 1',
        game_recorded_from: 'acc',
        map: 'Brands Hatch',
        car_name: 'BMW',
        userId: 'user-1',
        totalDataPoints: 100,
        totalChunks: 2,
        chunkSize: 50,
      },
    ]);
    expect(racingSessionModel.find).toHaveBeenCalledWith({ user_id: 'user-1' });
    expect(sort).toHaveBeenCalledWith({ created_date: -1, _id: -1 });
    expect(limit).toHaveBeenCalledWith(10);
  });

  it('initializes download for only the requested session when sessionId is provided', async () => {
    racingSessionModel.find.mockReturnValue({
      select: jest.fn().mockReturnValue({
        exec: jest.fn().mockResolvedValue([
          {
            _id: { toString: () => 'session-1' },
            session_name: 'Race 1',
            game_recorded_from: 'acc',
            map: 'Brands Hatch',
            car_name: 'BMW',
            user_id: 'user-1',
            totalDataPoints: 50,
            totalChunks: 2,
            dataChunkFileIds: ['file-1', 'file-2'],
          },
        ]),
      }),
    });

    const result = await service.initializeSessionsDownload('Brands Hatch', undefined, 1000, 'session-1');

    expect(result).toMatchObject({
      totalSessions: 1,
      totalChunks: 2,
      sessionMetadata: [
        {
          sessionId: 'session-1',
          session_name: 'Race 1',
          game_recorded_from: 'acc',
          map: 'Brands Hatch',
          car_name: 'BMW',
          userId: 'user-1',
          dataSize: 50,
          dataPoints: 50,
          chunkCount: 2,
        },
      ],
    });
    expect(result.downloadId).toEqual(expect.any(String));
    expect(racingSessionModel.find).toHaveBeenCalledWith({ _id: 'session-1', map: 'Brands Hatch' });
  });

  it('returns a stream for the requested download chunk', async () => {
    const sessionId = '507f1f77bcf86cd799439011';
    const stream = { pipe: jest.fn() };

    racingSessionModel.findById.mockReturnValue({
      select: jest.fn().mockReturnValue({
        exec: jest.fn().mockResolvedValue({
          dataChunkFileIds: ['507f1f77bcf86cd799439012', '507f1f77bcf86cd799439013'],
          totalDataPoints: 100,
        }),
      }),
    });
    gridfsService.downloadJSONStream.mockResolvedValue(stream);
    gridfsService.getFileSize.mockResolvedValue(4096);

    await expect(service.getSessionDownloadChunk(sessionId, 1)).resolves.toEqual({
      stream,
      fileSize: 4096,
      totalChunks: 2,
      dataPoints: 100,
    });
    expect(racingSessionModel.findById).toHaveBeenCalledWith(sessionId);
  });

  it('loads ordered telemetry chunks for segment classification', async () => {
    const sessionId = '507f1f77bcf86cd799439011';

    racingSessionModel.findById.mockReturnValue({
      select: jest.fn().mockReturnValue({
        exec: jest.fn().mockResolvedValue({
          map: 'Brands Hatch',
          car_name: 'BMW',
          user_id: 'user-1',
          dataChunkFileIds: ['507f1f77bcf86cd799439012', '507f1f77bcf86cd799439013'],
        }),
      }),
    });
    gridfsService.downloadJSON
      .mockResolvedValueOnce([{ row: 1 }])
      .mockResolvedValueOnce([{ row: 2 }, { row: 3 }]);

    await expect(service.getSessionTelemetryForClassification('user-1', sessionId)).resolves.toEqual({
      sessionId,
      trackName: 'Brands Hatch',
      carName: 'BMW',
      telemetryData: [{ row: 1 }, { row: 2 }, { row: 3 }],
    });
    expect(racingSessionModel.findById).toHaveBeenCalledWith(sessionId);
  });

  it('rejects segment classification telemetry owned by another user', async () => {
    const sessionId = '507f1f77bcf86cd799439011';

    racingSessionModel.findById.mockReturnValue({
      select: jest.fn().mockReturnValue({
        exec: jest.fn().mockResolvedValue({
          map: 'Brands Hatch',
          car_name: 'BMW',
          user_id: 'user-2',
          dataChunkFileIds: ['507f1f77bcf86cd799439012'],
        }),
      }),
    });

    await expect(
      service.getSessionTelemetryForClassification('user-1', sessionId),
    ).rejects.toThrow('Session not found or access denied');
    expect(gridfsService.downloadJSON).not.toHaveBeenCalled();
  });
});
