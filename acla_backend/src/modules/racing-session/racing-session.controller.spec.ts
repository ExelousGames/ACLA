import { Test, TestingModule } from '@nestjs/testing';
import { RacingSessionController } from './racing-session.controller';
import { RacingSessionService } from './racing-session.service';
import { UserSessionAiModelService } from '../user-session-ai-model/user-session-ai-model.service';
import { AiServiceClient } from 'src/shared/ai/ai-service.client';
import { UserInfoService } from '../user-info/user-info.service';
import { BadRequestException, ForbiddenException } from '@nestjs/common';

describe('RacingSessionController', () => {
  let controller: RacingSessionController;
  let racingSessionService: any;
  let aiServiceClient: any;

  beforeEach(async () => {
    racingSessionService = {
      listUserSessionsForAnalysis: jest.fn(),
      getSessionTelemetryForClassification: jest.fn(),
      createRacingSessionFromChunks: jest.fn(),
    };
    aiServiceClient = {
      classifySegments: jest.fn(),
      analyzeLiveRecordedAnalysis: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [RacingSessionController],
      providers: [
        { provide: RacingSessionService, useValue: racingSessionService },
        { provide: UserSessionAiModelService, useValue: {} },
        { provide: AiServiceClient, useValue: aiServiceClient },
        { provide: UserInfoService, useValue: {} },
      ],
    }).compile();

    controller = module.get<RacingSessionController>(RacingSessionController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  it.each([undefined, '', 'forza'])('rejects unsupported upload game %p before creating upload state', async (gameRecordedFrom) => {
    await expect(controller.initUpload({
      sessionName: 'Race 1',
      mapName: 'Monza',
      carName: 'GT3',
      userId: 'user-1',
      game_recorded_from: gameRecordedFrom as any,
    })).rejects.toBeInstanceOf(BadRequestException);

    expect((controller as any).uploadStates.size).toBe(0);
  });

  it.each(['acc', 'ac', 'iracing'] as const)('accepts supported upload game %s', async (gameRecordedFrom) => {
    await expect(controller.initUpload({
      sessionName: 'Race 1',
      mapName: 'Monza',
      carName: 'GT3',
      userId: 'user-1',
      game_recorded_from: gameRecordedFrom,
    })).resolves.toEqual({ uploadId: expect.any(String) });
  });

  it('persists upload game metadata when completing a chunked session', async () => {
    racingSessionService.createRacingSessionFromChunks.mockResolvedValue({ _id: 'session-1' });
    const { uploadId } = await controller.initUpload({
      sessionName: 'Race 1',
      mapName: 'Monza',
      carName: 'GT3',
      userId: 'user-1',
      game_recorded_from: 'acc',
    });

    await expect(controller.completeUpload({}, uploadId)).resolves.toMatchObject({
      sessionId: 'session-1',
    });
    expect(racingSessionService.createRacingSessionFromChunks).toHaveBeenCalledWith(
      'Race 1',
      'Monza',
      'GT3',
      'user-1',
      'acc',
      [],
      0,
      1000,
    );
  });

  it('blocks analysis metadata for another user', async () => {
    await expect(
      controller.initializeUserSessionsAnalysis(
        { user: { userId: 'user-1', username: 'driver@example.com' } },
        { userId: 'user-2' },
      ),
    ).rejects.toBeInstanceOf(ForbiddenException);
  });

  it('allows the AI service account to request analysis metadata for a target user', async () => {
    process.env.AI_SERVICE_USERNAME = 'ai@example.com';
    racingSessionService.listUserSessionsForAnalysis.mockResolvedValue([]);

    await expect(
      controller.initializeUserSessionsAnalysis(
        { user: { userId: 'service-user', username: 'ai@example.com' } },
        { userId: 'user-1', sessionLimit: 10 },
      ),
    ).resolves.toEqual({ userId: 'user-1', totalSessions: 0, sessions: [] });
    expect(racingSessionService.listUserSessionsForAnalysis).toHaveBeenCalledWith('user-1', 10);
  });

  it('loads and forwards a saved session for segment classification', async () => {
    racingSessionService.getSessionTelemetryForClassification.mockResolvedValue({
      sessionId: 'session-1',
      trackName: 'Brands Hatch',
      carName: 'BMW',
      telemetryData: [{ speed: 120 }],
    });
    aiServiceClient.classifySegments.mockResolvedValue({
      status: 'success',
      session_id: 'session-1',
      samples_analyzed: 1,
      parent_segment_count: 1,
      segments: [{
        id: 'segment-1',
        labels: ['EA'],
        track_section: 'brands_hatch2',
        start_index: 0,
        end_index: 1,
      }],
      expert_reference_data: [{
        raw_index: 4,
        expert_time_difference: 12,
        expert_optimal_player_pos_x: 100,
        expert_optimal_player_pos_y: 200,
        expert_optimal_player_pos_z: 300,
        Graphics_normalized_car_position: 0.4,
        expert_optimal_throttle: 0.8,
        expert_optimal_brake: 0.1,
        expert_optimal_gear: 4,
      }],
    });

    await expect(
      controller.classifySessionSegments(
        { user: { userId: 'user-1' } },
        { session_id: 'session-1' },
      ),
    ).resolves.toEqual({
      status: 'success',
      session_id: 'session-1',
      samples_analyzed: 1,
      parent_segment_count: 1,
      segments: [{
        id: 'segment-1',
        labels: ['EA'],
        track_section: 'brands_hatch2',
        start_index: 0,
        end_index: 1,
      }],
      expert_reference_data: [{
        raw_index: 4,
        expert_time_difference: 12,
        expert_optimal_player_pos_x: 100,
        expert_optimal_player_pos_y: 200,
        expert_optimal_player_pos_z: 300,
        Graphics_normalized_car_position: 0.4,
        expert_optimal_throttle: 0.8,
        expert_optimal_brake: 0.1,
        expert_optimal_gear: 4,
      }],
    });

    expect(racingSessionService.getSessionTelemetryForClassification).toHaveBeenCalledWith('user-1', 'session-1');
    expect(aiServiceClient.classifySegments).toHaveBeenCalledWith({
      session_id: 'session-1',
      telemetry_data: [{ speed: 120 }],
      track_name: 'Brands Hatch',
      car_name: 'BMW',
    });
  });

  it('forwards live baseline records for analysis', async () => {
    aiServiceClient.analyzeLiveRecordedAnalysis.mockResolvedValue({
      status: 'success',
      session_id: 'live-baseline-lap-2',
      samples_analyzed: 1,
      parent_segment_count: 0,
      segments: [],
      expert_time_available: false,
      expert_reference_data: [{
        raw_index: 1,
        expert_time_difference: 8,
        expert_optimal_player_pos_x: 110,
        expert_optimal_player_pos_y: 210,
        expert_optimal_player_pos_z: 310,
        Graphics_normalized_car_position: 0.5,
        expert_optimal_throttle: 0.9,
        expert_optimal_brake: 0,
        expert_optimal_gear: 5,
      }],
    });

    await expect(
      controller.analyzeLiveRecordedAnalysis(
        { user: { userId: 'user-1' } },
        {
          track: 'brands_hatch',
          car: 'Ferrari 296',
          baseline_lap: 2,
          records: [{ speed: 120 }],
        },
      ),
    ).resolves.toEqual({
      status: 'success',
      session_id: 'live-baseline-lap-2',
      samples_analyzed: 1,
      parent_segment_count: 0,
      segments: [],
      expert_time_available: false,
      expert_reference_data: [{
        raw_index: 1,
        expert_time_difference: 8,
        expert_optimal_player_pos_x: 110,
        expert_optimal_player_pos_y: 210,
        expert_optimal_player_pos_z: 310,
        Graphics_normalized_car_position: 0.5,
        expert_optimal_throttle: 0.9,
        expert_optimal_brake: 0,
        expert_optimal_gear: 5,
      }],
    });

    expect(aiServiceClient.analyzeLiveRecordedAnalysis).toHaveBeenCalledWith({
      track: 'brands_hatch',
      car: 'Ferrari 296',
      baseline_lap: 2,
      records: [{ speed: 120 }],
    });
  });

  it('rejects empty live baseline records', async () => {
    await expect(
      controller.analyzeLiveRecordedAnalysis(
        { user: { userId: 'user-1' } },
        {
          track: 'brands_hatch',
          car: 'Ferrari 296',
          baseline_lap: 2,
          records: [],
        },
      ),
    ).rejects.toThrow('records is required');

    expect(aiServiceClient.analyzeLiveRecordedAnalysis).not.toHaveBeenCalled();
  });
});
