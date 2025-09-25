import { Test, TestingModule } from '@nestjs/testing';
import { AiModelController } from './ai-model.controller';
import { AiModelService } from './ai-model.service';
import { ChunkClientService } from '../../shared/chunk-service/chunk.service';
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';
import { ObjectId } from 'mongodb';

describe('AiModelController - Chunked Data Transfer', () => {
    let controller: AiModelController;
    let aiModelService: jest.Mocked<AiModelService>;
    let chunkService: jest.Mocked<ChunkClientService>;
    let gridfsService: jest.Mocked<GridFSService>;

    const mockModelDoc = {
        _id: new ObjectId(),
        trackName: 'brands_hatch',
        carName: 'AllCars',
        modelType: 'imitation_learning',
        isActive: true,
        modelDataFileId: new ObjectId(),
        metadata: {},
    } as any; // Use 'as any' to bypass strict typing for tests

    beforeEach(async () => {
        const mockAiModelService = {
            getActiveModel: jest.fn(),
            create: jest.fn(),
            findAll: jest.fn(),
            findOne: jest.fn(),
            findOneWithModelData: jest.fn(),
            update: jest.fn(),
            remove: jest.fn(),
            activateModel: jest.fn(),
            save_ai_model: jest.fn(),
            save_ai_model_from_file: jest.fn(),
            isGridFSHealthy: jest.fn(),
            getGridFSStats: jest.fn(),
            getGridFSServiceInfo: jest.fn(),
            reinitializeGridFS: jest.fn(),
        };

        const mockChunkService = {
            handleIncomingChunk: jest.fn(),
            getSessionStatus: jest.fn(),
            getPreparedChunk: jest.fn(),
        };

        const mockGridfsService = {
            getFileSize: jest.fn(),
            getFileInfo: jest.fn(),
            downloadFileRange: jest.fn(),
            uploadJSON: jest.fn(),
            downloadJSON: jest.fn(),
            deleteFile: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [AiModelController],
            providers: [
                {
                    provide: AiModelService,
                    useValue: mockAiModelService,
                },
                {
                    provide: ChunkClientService,
                    useValue: mockChunkService,
                },
                {
                    provide: GridFSService,
                    useValue: mockGridfsService,
                },
            ],
        }).compile();

        controller = module.get<AiModelController>(AiModelController);
        aiModelService = module.get(AiModelService);
        chunkService = module.get(ChunkClientService);
        gridfsService = module.get(GridFSService);
    });

    describe('initGetActiveModelData', () => {
        it('should prepare chunked data for a small file successfully', async () => {
            // Arrange
            const modelType = 'imitation_learning';
            const trackName = 'brands_hatch';
            const carName = 'AllCars';
            const fileSize = 5 * 1024 * 1024; // 5MB (not large)

            aiModelService.getActiveModel.mockResolvedValue(mockModelDoc);
            gridfsService.getFileSize.mockResolvedValue(fileSize);
            gridfsService.getFileInfo.mockResolvedValue({
                filename: 'test-model.json',
                length: fileSize,
            });

            // Act
            const result = await controller.initGetActiveModelData(modelType, trackName, carName);

            // Assert
            expect(result.success).toBe(true);
            expect(result.sessionId).toMatch(/^gridfs_\d+_/);
            expect(result.totalChunks).toBeGreaterThan(0);
            expect(result.isLargeFile).toBe(false);
            expect(result.fileSize).toBe('5 MB');
            expect(result.modelInfo).toEqual({
                trackName: mockModelDoc.trackName,
                carName: mockModelDoc.carName,
                modelType: mockModelDoc.modelType,
                isActive: mockModelDoc.isActive,
            });

            expect(aiModelService.getActiveModel).toHaveBeenCalledWith(trackName, carName, modelType);
            expect(gridfsService.getFileSize).toHaveBeenCalledWith(mockModelDoc.modelDataFileId, GRIDFS_BUCKETS.AI_MODELS);
        });

        it('should prepare chunked data for a large file successfully', async () => {
            // Arrange
            const modelType = 'imitation_learning';
            const trackName = 'brands_hatch';
            const carName = 'AllCars';
            const fileSize = 50 * 1024 * 1024; // 50MB (large file)

            aiModelService.getActiveModel.mockResolvedValue(mockModelDoc);
            gridfsService.getFileSize.mockResolvedValue(fileSize);
            gridfsService.getFileInfo.mockResolvedValue({
                filename: 'test-model.json',
                length: fileSize,
            });

            // Act
            const result = await controller.initGetActiveModelData(modelType, trackName, carName);

            // Assert
            expect(result.success).toBe(true);
            expect(result.sessionId).toMatch(/^gridfs_\d+_/);
            expect(result.totalChunks).toBeGreaterThan(0);
            expect(result.isLargeFile).toBe(true);
            expect(result.fileSize).toBe('50 MB');
            expect(result.modelInfo).toEqual({
                trackName: mockModelDoc.trackName,
                carName: mockModelDoc.carName,
                modelType: mockModelDoc.modelType,
                isActive: mockModelDoc.isActive,
            });
        });

        it('should return error when no active model is found', async () => {
            // Arrange
            const modelType = 'imitation_learning';
            const trackName = 'brands_hatch';
            const carName = 'AllCars';

            aiModelService.getActiveModel.mockResolvedValue(null as any);

            // Act
            const result = await controller.initGetActiveModelData(modelType, trackName, carName);

            // Assert
            expect(result.success).toBe(false);
            expect(result.message).toBe('No active model found for the specified criteria');
            expect(aiModelService.getActiveModel).toHaveBeenCalledWith(trackName, carName, modelType);
        });

        it('should return error when model has no data file', async () => {
            // Arrange
            const modelType = 'imitation_learning';
            const trackName = 'brands_hatch';
            const carName = 'AllCars';
            const modelDocWithoutFile = { ...mockModelDoc, modelDataFileId: null } as any;

            aiModelService.getActiveModel.mockResolvedValue(modelDocWithoutFile);

            // Act
            const result = await controller.initGetActiveModelData(modelType, trackName, carName);

            // Assert
            expect(result.success).toBe(false);
            expect(result.message).toBe('Model found but no data file is associated with it');
        });

        it('should handle errors gracefully', async () => {
            // Arrange
            const modelType = 'imitation_learning';
            const trackName = 'brands_hatch';
            const carName = 'AllCars';
            const error = new Error('Database connection failed');

            aiModelService.getActiveModel.mockRejectedValue(error);

            // Act
            const result = await controller.initGetActiveModelData(modelType, trackName, carName);

            // Assert
            expect(result.success).toBe(false);
            expect(result.message).toBe('Failed to prepare chunked data: Database connection failed');
            expect(result.recommendation).toBe('Check server logs for detailed error information.');
        });

        it('should work without optional query parameters', async () => {
            // Arrange
            const modelType = 'imitation_learning';
            const fileSize = 1024 * 1024; // 1MB

            aiModelService.getActiveModel.mockResolvedValue(mockModelDoc);
            gridfsService.getFileSize.mockResolvedValue(fileSize);
            gridfsService.getFileInfo.mockResolvedValue({
                filename: 'test-model.json',
                length: fileSize,
            });

            // Act
            const result = await controller.initGetActiveModelData(modelType);

            // Assert
            expect(result.success).toBe(true);
            expect(aiModelService.getActiveModel).toHaveBeenCalledWith(undefined, undefined, modelType);
        });
    });

    describe('getActiveModelDataChunk', () => {
        const mockSessionId = 'gridfs_1234567890_abcdef123';
        const mockChunkData = 'test chunk data string';

        beforeEach(() => {
            // Setup streaming session metadata for tests
            const sessionMetadata = {
                fileId: mockModelDoc.modelDataFileId,
                bucketName: GRIDFS_BUCKETS.AI_MODELS,
                totalChunks: 5,
                chunkSize: 512 * 1024,
                fileSize: 2.5 * 1024 * 1024,
                filename: 'test-model.json',
                createdAt: Date.now(),
            };

            // Use reflection to access private property
            (controller as any).streamingSessions = new Map();
            (controller as any).streamingSessions.set(mockSessionId, sessionMetadata);
        });

        it('should retrieve a streaming chunk successfully', async () => {
            // Arrange
            const chunkIndex = '0';

            gridfsService.downloadFileRange.mockResolvedValue(mockChunkData);

            // Act
            const result = await controller.getActiveModelDataChunk(mockSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(true);
            expect(result.sessionId).toBe(mockSessionId);
            expect(result.chunkIndex).toBe(0);
            expect(result.totalChunks).toBe(5);
            expect(result.data).toBe(mockChunkData);
            expect(result.isLastChunk).toBe(false);
            expect((result as any).byteRange).toBe('0-524287/2621440');

            expect(gridfsService.downloadFileRange).toHaveBeenCalledWith(
                mockModelDoc.modelDataFileId,
                GRIDFS_BUCKETS.AI_MODELS,
                0,
                524287
            );
        });

        it('should retrieve the last chunk correctly', async () => {
            // Arrange
            const chunkIndex = '4'; // Last chunk (0-indexed)

            gridfsService.downloadFileRange.mockResolvedValue(mockChunkData);

            // Act
            const result = await controller.getActiveModelDataChunk(mockSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(true);
            expect(result.isLastChunk).toBe(true);
            expect(result.chunkIndex).toBe(4);
        });

        it('should return error for invalid chunk index (negative)', async () => {
            // Arrange
            const chunkIndex = '-1';

            // Act
            const result = await controller.getActiveModelDataChunk(mockSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(false);
            expect((result as any).message).toBe('Invalid chunk index');
        });

        it('should return error for invalid chunk index (non-numeric)', async () => {
            // Arrange
            const chunkIndex = 'invalid';

            // Act
            const result = await controller.getActiveModelDataChunk(mockSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(false);
            expect((result as any).message).toBe('Invalid chunk index');
        });

        it('should return error for chunk index out of bounds', async () => {
            // Arrange
            const chunkIndex = '10'; // Beyond totalChunks (5)

            // Act
            const result = await controller.getActiveModelDataChunk(mockSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(false);
            expect((result as any).message).toBe('Failed to retrieve streaming chunk: Invalid chunk index. Must be between 0 and 4');
        });

        it('should return error for expired/non-existent session', async () => {
            // Arrange
            const nonExistentSessionId = 'gridfs_nonexistent_session';
            const chunkIndex = '0';

            // Act
            const result = await controller.getActiveModelDataChunk(nonExistentSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(false);
            expect((result as any).message).toBe('Failed to retrieve streaming chunk: Session not found or expired');
        });

        it('should handle GridFS read errors gracefully', async () => {
            // Arrange
            const chunkIndex = '0';
            const error = new Error('GridFS read failed');

            gridfsService.downloadFileRange.mockRejectedValue(error);

            // Act
            const result = await controller.getActiveModelDataChunk(mockSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(false);
            expect((result as any).message).toBe('Failed to retrieve streaming chunk: Failed to read chunk 0: GridFS read failed');
        });

        it('should fallback to regular chunk service for non-streaming sessions', async () => {
            // Arrange
            const regularSessionId = 'regular_session_123';
            const chunkIndex = '0';
            const mockChunkResult = {
                success: true,
                sessionId: regularSessionId,
                chunkIndex: 0,
                totalChunks: 3,
                data: 'test data',
                isLastChunk: false,
            };

            const mockSessionStatus = {
                success: true,
                sessionId: regularSessionId,
                status: 'complete' as const,
                totalChunks: 3,
                receivedChunks: 3,
                completionPercentage: 100,
                createdAt: new Date(),
                lastUpdated: new Date(),
            };

            chunkService.getSessionStatus.mockResolvedValue(mockSessionStatus);
            chunkService.getPreparedChunk.mockResolvedValue(mockChunkResult);

            // Act
            const result = await controller.getActiveModelDataChunk(regularSessionId, chunkIndex);

            // Assert
            expect(result).toEqual(mockChunkResult);
            expect(chunkService.getSessionStatus).toHaveBeenCalledWith(regularSessionId);
            expect(chunkService.getPreparedChunk).toHaveBeenCalledWith(regularSessionId, 0);
        });

        it('should return error when regular chunk session is not found', async () => {
            // Arrange
            const regularSessionId = 'regular_session_123';
            const chunkIndex = '0';

            const mockSessionStatus = {
                success: false,
                sessionId: regularSessionId,
                status: 'failed' as const,
                totalChunks: 0,
                receivedChunks: 0,
                completionPercentage: 0,
                createdAt: new Date(),
                lastUpdated: new Date(),
            };

            chunkService.getSessionStatus.mockResolvedValue(mockSessionStatus);

            // Act
            const result = await controller.getActiveModelDataChunk(regularSessionId, chunkIndex);

            // Assert
            expect(result.success).toBe(false);
            expect((result as any).message).toBe('Session not found or expired');
        });
    });

    describe('Integration Tests', () => {
        it('should handle complete chunked data flow', async () => {
            // Arrange
            const modelType = 'imitation_learning';
            const trackName = 'brands_hatch';
            const carName = 'AllCars';
            const fileSize = 1536 * 1024; // 1.5MB (3 chunks of 512KB each)

            aiModelService.getActiveModel.mockResolvedValue(mockModelDoc);
            gridfsService.getFileSize.mockResolvedValue(fileSize);
            gridfsService.getFileInfo.mockResolvedValue({
                filename: 'test-model.json',
                length: fileSize,
            });

            // Act - Step 1: Prepare chunked data
            const prepareResult = await controller.initGetActiveModelData(modelType, trackName, carName);

            // Assert preparation
            expect(prepareResult.success).toBe(true);
            expect(prepareResult.totalChunks).toBe(3);

            // Act - Step 2: Retrieve chunks
            const chunk1Data = 'chunk 1 data';
            const chunk2Data = 'chunk 2 data';
            const chunk3Data = 'chunk 3 data';

            gridfsService.downloadFileRange
                .mockResolvedValueOnce(chunk1Data)
                .mockResolvedValueOnce(chunk2Data)
                .mockResolvedValueOnce(chunk3Data);

            const chunk1Result = await controller.getActiveModelDataChunk(prepareResult.sessionId, '0');
            const chunk2Result = await controller.getActiveModelDataChunk(prepareResult.sessionId, '1');
            const chunk3Result = await controller.getActiveModelDataChunk(prepareResult.sessionId, '2');

            // Assert chunk retrieval
            expect(chunk1Result.success).toBe(true);
            expect(chunk1Result.isLastChunk).toBe(false);
            expect(chunk1Result.data).toBe(chunk1Data);

            expect(chunk2Result.success).toBe(true);
            expect(chunk2Result.isLastChunk).toBe(false);
            expect(chunk2Result.data).toBe(chunk2Data);

            expect(chunk3Result.success).toBe(true);
            expect(chunk3Result.isLastChunk).toBe(true);
            expect(chunk3Result.data).toBe(chunk3Data);
        });
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });
});