import { Test, TestingModule } from '@nestjs/testing';
import { AiModelService } from './ai-model.service';

describe('AiModelService', () => {
    let service: AiModelService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [AiModelService],
        }).compile();

        service = module.get<AiModelService>(AiModelService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });
});
