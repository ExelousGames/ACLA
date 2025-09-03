import { Test, TestingModule } from '@nestjs/testing';
import { UserSessionAiModelController } from './user-session-ai-model.controller';
import { UserSessionAiModelService } from './user-session-ai-model.service';

describe('AiModelController', () => {
    let controller: UserSessionAiModelController;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [UserSessionAiModelController],
            providers: [UserSessionAiModelService],
        }).compile();

        controller = module.get<UserSessionAiModelController>(UserSessionAiModelController);
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });
});
