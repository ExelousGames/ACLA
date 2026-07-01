import { Test, TestingModule } from '@nestjs/testing';
import { UserInfoService } from './user-info.service';
import { getModelToken } from '@nestjs/mongoose';
import { UserInfo } from '../../schemas/user-info.schema';
import { PasswordService } from 'src/shared/utils/password.service';

describe('UserInfoService', () => {
  let service: UserInfoService;
  let userInfoModel: {
    findOne: jest.Mock;
    findOneAndUpdate: jest.Mock;
  };

  beforeEach(async () => {
    userInfoModel = {
      findOne: jest.fn(),
      findOneAndUpdate: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        UserInfoService,
        { provide: getModelToken(UserInfo.name), useValue: userInfoModel },
        { provide: PasswordService, useValue: { hashPassword: jest.fn() } },
      ],
    }).compile();

    service = module.get<UserInfoService>(UserInfoService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('gets a saved user summary', async () => {
    userInfoModel.findOne.mockReturnValue({
      exec: jest.fn().mockResolvedValue({ userSummary: { pace: 'steady' } }),
    });

    await expect(service.getUserSummary('user-1')).resolves.toEqual({ pace: 'steady' });
    expect(userInfoModel.findOne).toHaveBeenCalledWith({ id: 'user-1' });
  });

  it('returns an empty summary when none is saved', async () => {
    userInfoModel.findOne.mockReturnValue({
      exec: jest.fn().mockResolvedValue(null),
    });

    await expect(service.getUserSummary('user-1')).resolves.toEqual({});
  });

  it('updates a user summary', async () => {
    userInfoModel.findOneAndUpdate.mockReturnValue({
      exec: jest.fn().mockResolvedValue({ userSummary: { braking: 'late' } }),
    });

    await expect(service.updateUserSummary('user-1', { braking: 'late' })).resolves.toEqual({
      braking: 'late',
    });
    expect(userInfoModel.findOneAndUpdate).toHaveBeenCalledWith(
      { id: 'user-1' },
      { userSummary: { braking: 'late' } },
      { new: true },
    );
  });
});
