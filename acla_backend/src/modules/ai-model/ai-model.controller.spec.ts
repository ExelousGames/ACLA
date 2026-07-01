import { Test, TestingModule } from '@nestjs/testing';
import { AiModelController } from './ai-model.controller';
import { AiModelService } from './ai-model.service';
import { ChunkClientService } from '../../shared/chunk-service/chunk.service';
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';
import { ObjectId } from 'mongodb';

describe('AiModelController', () => {
    let controller: AiModelController;
    let aiModelService: jest.Mocked<AiModelService>;
    let chunkService: jest.Mocked<ChunkClientService>;
    let gridfsService: jest.Mocked<GridFSService>;

    const mockModelDoc = {
        _id: new ObjectId(),
        modelType: 'imitation_learning',
        isActive: true,
        modelDataFileId: new ObjectId(),
        metadata: { foo: 'bar' },
    } as any;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [AiModelController],
            providers: [
                {
                    provide: AiModelService,
                    useValue: {
                        create: jest.fn(),
                        findAll: jest.fn(),
                        findOne: jest.fn(),
                        findOneWithModelData: jest.fn(),
                        update: jest.fn(),
                        remove: jest.fn(),
                        getActiveModel: jest.fn(),
                        getActiveModelWithData: jest.fn(),
                        activateModel: jest.fn(),
                        save_ai_model: jest.fn(),
                        save_ai_model_from_file: jest.fn(),
                        isGridFSHealthy: jest.fn(),
                        getGridFSStats: jest.fn(),
                        getGridFSServiceInfo: jest.fn(),
                        reinitializeGridFS: jest.fn(),
                    },
                },
                {
                    provide: ChunkClientService,
                    useValue: {
                        handleIncomingChunk: jest.fn(),
                        getSessionStatus: jest.fn(),
                        getPreparedChunk: jest.fn(),
                    },
                },
                {
                    provide: GridFSService,
                    useValue: {
                        getFileSize: jest.fn(),
                        downloadFileRangeAsBase64: jest.fn(),
                        uploadJSON: jest.fn(),
                        downloadJSON: jest.fn(),
                        deleteFile: jest.fn(),
                        getAllBucketsStats: jest.fn(),
                        isHealthy: jest.fn(),
                        getServiceInfo: jest.fn(),
                        reinitialize: jest.fn(),
                    },
                },
            ],
        }).compile();

        controller = module.get(AiModelController);
        aiModelService = module.get(AiModelService) as jest.Mocked<AiModelService>;
        chunkService = module.get(ChunkClientService) as jest.Mocked<ChunkClientService>;
        gridfsService = module.get(GridFSService) as jest.Mocked<GridFSService>;

        jest.clearAllMocks();
    });

    describe('initGetActiveModelData', () => {
        it('returns chunking metadata when model is available', async () => {
            const fileSize = 1024 * 1024;
            aiModelService.getActiveModel.mockResolvedValue(mockModelDoc);
            gridfsService.getFileSize.mockResolvedValue(fileSize);

            const result = await controller.initGetActiveModelData('imitation_learning');

            expect(result.success).toBe(true);
            expect(aiModelService.getActiveModel).toHaveBeenCalledWith('imitation_learning');
            expect(result.data).toMatchObject({
                modelType: mockModelDoc.modelType,
                isActive: mockModelDoc.isActive,
                metadata: mockModelDoc.metadata,
                modelData: null,
            });
            expect(result.chunking).toBeDefined();
            expect(result.chunking?.chunkSize).toBe(512 * 1024);
            expect(result.chunking?.totalChunks).toBe(Math.ceil(fileSize / (512 * 1024)));
            const sessionId = result.chunking?.sessionId;
            expect(sessionId).toBeDefined();
            expect((controller as any).streamingSessions.has(sessionId as string)).toBe(true);
        });

        it('returns error when model has no data file', async () => {
            aiModelService.getActiveModel.mockResolvedValue({
                ...mockModelDoc,
                modelDataFileId: null,
            });

            const result = await controller.initGetActiveModelData('imitation_learning');

            expect(result.success).toBe(false);
            expect(result.message).toBe('Model found but no data file is associated with it');
        });

        it('handles service errors gracefully', async () => {
            aiModelService.getActiveModel.mockRejectedValue(new Error('boom'));

            const result = await controller.initGetActiveModelData('imitation_learning');

            expect(result.success).toBe(false);
            expect(result.message).toBe('Failed to initialize chunked data: boom');
        });
    });

    describe('getActiveModelDataChunk', () => {
        const sessionId = 'session_123';
        const baseMetadata = {
            fileId: mockModelDoc.modelDataFileId,
            bucketName: GRIDFS_BUCKETS.AI_MODELS,
            totalChunks: 2,
            chunkSize: 512 * 1024,
            fileSize: 768 * 1024,
            filename: 'imitation_learning',
            createdAt: Date.now(),
        };

        beforeEach(() => {
            (controller as any).streamingSessions = new Map();
            (controller as any).streamingSessions.set(sessionId, baseMetadata);
        });

        it('returns chunk data for a valid request', async () => {
            gridfsService.downloadFileRangeAsBase64.mockResolvedValue('chunk-data');

            const result = await controller.getActiveModelDataChunk(sessionId, '0');

            expect(result.success).toBe(true);
            expect(result.data).toBe('chunk-data');
            expect(result.chunkIndex).toBe(0);
            expect(gridfsService.downloadFileRangeAsBase64).toHaveBeenCalledWith(
                expect.any(ObjectId),
                baseMetadata.bucketName,
                0,
                baseMetadata.chunkSize,
            );
        });

        it('rejects invalid indices', async () => {
            const result = await controller.getActiveModelDataChunk(sessionId, '-1');
            expect(result.success).toBe(false);
            expect(result.message).toBe('Invalid chunk index');
        });

        it('handles missing sessions', async () => {
            const result = await controller.getActiveModelDataChunk('missing', '0');
            expect(result.success).toBe(false);
            expect(result.message).toBe('Failed to retrieve chunk: Session not found or expired');
        });

        it('handles GridFS read errors gracefully', async () => {
            gridfsService.downloadFileRangeAsBase64.mockRejectedValue(new Error('read error'));

            const result = await controller.getActiveModelDataChunk(sessionId, '0');

            expect(result.success).toBe(false);
            expect(result.message).toBe('Failed to retrieve chunk: Failed to read chunk 0: read error');
        });
    });
});