import { Test, TestingModule } from '@nestjs/testing';
import { UserSessionAiModelService } from './user-session-ai-model.service';

describe('AiModelService', () => {
    let service: UserSessionAiModelService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [UserSessionAiModelService],
        }).compile();

        service = module.get<UserSessionAiModelService>(UserSessionAiModelService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });
});
