

import { Controller, Get, Post, Put, Delete, Body, Param, Query } from '@nestjs/common';
import { AiModelService } from './ai-model.service';
import { CreateAiModelDto, UpdateAiModelDto } from './dto/ai-model.dto';
import { ChunkClientService } from '../../shared/chunk-service/chunk.service';
import { ChunkData } from '../../shared/chunk-service/interfaces/chunk.interface';
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';
import { ObjectId } from 'mongodb';
import { promises } from 'dns';

@Controller('ai-model')
export class AiModelController {
    constructor(
        private readonly aiModelService: AiModelService,
        private readonly chunkService: ChunkClientService,
        private readonly gridfsService: GridFSService
    ) { }

    @Post()
    async create(@Body() createAiModelDto: CreateAiModelDto) {
        return this.aiModelService.create(createAiModelDto);
    }

    @Get()
    async findAll(
        @Query('trackName') trackName?: string,
        @Query('carName') carName?: string,
        @Query('modelType') modelType?: string,
        @Query('isActive') isActive?: boolean
    ) {
        return this.aiModelService.findAll({ trackName, carName, modelType, isActive });
    }

    @Get(':id')
    async findOne(@Param('id') id: string) {
        return this.aiModelService.findOne(id);
    }

    @Get(':id/with-data')
    async findOneWithData(@Param('id') id: string) {
        try {
            const result = await this.aiModelService.findOneWithModelData(id);

            // If the model data is too large, provide helpful guidance
            if (result.modelDataInfo?.isLargeFile) {
                return {
                    ...result,
                    _guidance: {
                        message: 'This model contains large data that cannot be returned in a single response.',
                        recommendedEndpoints: {
                            prepareChunked: `/api/ai-model/active/${result.modelType}/prepare-chunked?trackName=${result.trackName}&carName=${result.carName}`,
                            getChunk: '/api/ai-model/active/chunked-data/{sessionId}/{chunkIndex}'
                        },
                        steps: [
                            '1. Call the prepare-chunked endpoint to get a session ID',
                            '2. Use the session ID to fetch individual chunks',
                            '3. Combine chunks on the client side to reconstruct the full data'
                        ]
                    }
                };
            }

            return result;
        } catch (error) {
            console.error(`Error in findOneWithData: ${error.message}`);
            return {
                success: false,
                message: `Failed to retrieve model data: ${error.message}`,
                _guidance: {
                    message: 'If this is a large model, consider using the chunked data endpoints.',
                    recommendedEndpoints: {
                        prepareChunked: '/api/ai-model/active/{modelType}/prepare-chunked',
                        getChunk: '/api/ai-model/active/chunked-data/{sessionId}/{chunkIndex}'
                    }
                }
            };
        }
    }

    @Put(':id')
    async update(@Param('id') id: string, @Body() updateAiModelDto: UpdateAiModelDto) {
        return this.aiModelService.update(id, updateAiModelDto);
    }

    @Delete(':id')
    async remove(@Param('id') id: string) {
        return this.aiModelService.remove(id);
    }

    /**
     * Initialize chunked retrieval of active model data.
     * Returns the final structure with metadata and chunking information.
     * @param modelType Required - The type of model to retrieve
     * @param trackName Optional - The track name to filter by (query parameter)
     * @param carName Optional - The car name to filter by (query parameter)
     * @returns Final structure with metadata and chunking session info
     */
    @Get('active/:modelType/prepare-chunked')
    async initGetActiveModelData(
        @Param('modelType') modelType: string,
        @Query('trackName') trackName?: string,
        @Query('carName') carName?: string
    ) {
        console.log(`Initializing chunked model data - Track: ${trackName}, Car: ${carName}, Type: ${modelType}`);

        try {
            // Get model metadata
            const modelDoc = await this.aiModelService.getActiveModel(trackName, carName, modelType);

            if (!modelDoc) {
                return {
                    success: false,
                    message: 'No active model found for the specified criteria'
                };
            }

            if (!modelDoc.modelDataFileId) {
                return {
                    success: false,
                    message: 'Model found but no data file is associated with it'
                };
            }

            // Get file info
            const fileSize = await this.gridfsService.getFileSize(new ObjectId(modelDoc.modelDataFileId), GRIDFS_BUCKETS.AI_MODELS);
            const chunkSize = 512 * 1024; // 512KB chunks
            const totalChunks = Math.ceil(fileSize / chunkSize);

            // Create session for chunked transfer
            const sessionId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

            // Store session metadata
            this.storeStreamingSessionMetadata(sessionId, {
                fileId: new ObjectId(modelDoc.modelDataFileId),
                bucketName: GRIDFS_BUCKETS.AI_MODELS,
                totalChunks: totalChunks,
                chunkSize: chunkSize,
                fileSize: fileSize,
                filename: `${modelType}_${trackName}_${carName}`,
                createdAt: Date.now()
            });

            console.log(`Prepared chunked transfer: ${totalChunks} chunks of ${this.formatFileSize(chunkSize)} each`);

            // Return final structure with metadata and chunking info
            return {
                success: true,
                data: {
                    // Model metadata (available immediately)
                    modelType: modelDoc.modelType,
                    trackName: modelDoc.trackName,
                    carName: modelDoc.carName,
                    isActive: modelDoc.isActive,
                    metadata: modelDoc.metadata || {},

                    // Placeholder for chunked data (will be filled by client)
                    modelData: null // This will be populated from chunks
                },
                chunking: {
                    sessionId: sessionId,
                    totalChunks: totalChunks,
                    chunkSize: chunkSize,
                    totalSize: fileSize,
                    totalSizeHuman: this.formatFileSize(fileSize),
                    fieldsToFill: ['modelData'] // Fields that will be populated from chunks
                },
                message: `Model structure ready, ${totalChunks} chunks to retrieve`
            };
        } catch (error) {
            console.error(`Error initializing chunked data: ${error.message}`);
            return {
                success: false,
                message: `Failed to initialize chunked data: ${error.message}`
            };
        }
    }

    /**
     * Helper method to format file size in human-readable format
     */
    private formatFileSize(bytes: number): string {
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    /**
     * Get a specific chunk of model data.
     * @param sessionId The session ID from prepare-chunked endpoint
     * @param chunkIndex The index of the chunk to retrieve (0-based)
     * @returns Raw chunk data as base64 string
     */
    @Get('active/chunked-data/:sessionId/:chunkIndex')
    async getActiveModelDataChunk(
        @Param('sessionId') sessionId: string,
        @Param('chunkIndex') chunkIndex: string
    ) {
        const chunkIndexNum = parseInt(chunkIndex, 10);

        if (isNaN(chunkIndexNum) || chunkIndexNum < 0) {
            return {
                success: false,
                message: 'Invalid chunk index'
            };
        }

        try {
            const streamingChunk = await this.getStreamingChunk(sessionId, chunkIndexNum);
            return {
                success: true,
                sessionId,
                chunkIndex: chunkIndexNum,
                totalChunks: streamingChunk.totalChunks,
                data: streamingChunk.data, // Base64 encoded chunk data
                isLastChunk: streamingChunk.isLastChunk
            };
        } catch (error) {
            console.error(`Error getting chunk ${chunkIndexNum}: ${error.message}`);
            return {
                success: false,
                message: `Failed to retrieve chunk: ${error.message}`
            };
        }
    }

    @Post(':id/activate')
    async activateModel(@Param('id') id: string) {
        return this.aiModelService.activateModel(id);
    }

    @Post('save')
    async save_one_ai_model(@Body() chunkData: ChunkData) {
        try {
            return this.chunkService.handleIncomingChunk(
                chunkData,
                async (assembledPathOrData: any) => {
                    // If streaming/file mode is used, assembledPathOrData will be a file path.
                    // The service expects UpdateAiModelDto with modelData; for very large JSON, parse in a streaming manner if possible.
                    // For simplicity here, we will load from file in chunks and JSON.parse at the service layer if needed.
                    // However, to avoid memory spikes, let the service provide a file-based saver.
                    if (typeof assembledPathOrData === 'string') {
                        // Provide the file path to service to handle streaming upload to GridFS
                        return this.aiModelService.save_ai_model_from_file(
                            assembledPathOrData
                        );
                    }
                    // Legacy small payload path (assembled JSON object)
                    return this.aiModelService.save_ai_model(assembledPathOrData as UpdateAiModelDto);
                },
                { assemblyMode: 'file' }
            );
        } catch (error) {
            console.error(`Error saving AI model from file: ${error.message}`);
            return {
                success: false,
                message: `Failed to save AI model from file: ${error.message}`
            };
        }

    }

    // New GridFS-related endpoints
    @Get('gridfs/health')
    async checkGridFSHealth() {
        return {
            healthy: await this.aiModelService.isGridFSHealthy(),
            timestamp: new Date().toISOString()
        };
    }

    @Get('gridfs/stats')
    async getGridFSStats() {
        return this.aiModelService.getGridFSStats();
    }

    @Get('gridfs/info')
    async getGridFSInfo() {
        return this.aiModelService.getGridFSServiceInfo();
    }

    @Post('gridfs/reinitialize')
    async reinitializeGridFS() {
        try {
            await this.aiModelService.reinitializeGridFS();
            return { message: 'GridFS reinitialized successfully' };
        } catch (error) {
            return {
                message: 'Failed to reinitialize GridFS',
                error: error.message
            };
        }
    }

    /**
     * Store streaming session metadata for later chunk retrieval
     * In a production system, this should use Redis or a proper session store
     */
    private streamingSessions: Map<string, any> = new Map();

    private storeStreamingSessionMetadata(sessionId: string, metadata: any): void {
        this.streamingSessions.set(sessionId, metadata);

        // Set a timeout to clean up the session after 15 minutes
        setTimeout(() => {
            this.streamingSessions.delete(sessionId);
            console.log(`Cleaned up streaming session: ${sessionId}`);
        }, 15 * 60 * 1000); // 15 minutes
    }

    /**
     * Get streaming chunk for a session by reading directly from GridFS
     * Returns chunk data as base64 for safe transport
     */
    private async getStreamingChunk(sessionId: string, chunkIndex: number): Promise<any> {
        const sessionMetadata = this.streamingSessions.get(sessionId);
        if (!sessionMetadata) {
            throw new Error('Session not found or expired');
        }

        const { fileId, bucketName, totalChunks, chunkSize, fileSize } = sessionMetadata;

        if (chunkIndex < 0 || chunkIndex >= totalChunks) {
            throw new Error(`Invalid chunk index. Must be between 0 and ${totalChunks - 1}`);
        }

        try {
            // Calculate the byte range for this chunk
            // Note: GridFS end parameter is exclusive, so we don't subtract 1
            const startByte = chunkIndex * chunkSize;
            const endByte = Math.min(startByte + chunkSize, fileSize);
            const isLastChunk = chunkIndex === totalChunks - 1;

            // Debug last few chunks
            if (chunkIndex >= totalChunks - 3) {
                console.log(`🔍 DEBUG Last chunk ${chunkIndex}: startByte=${startByte}, endByte=${endByte}, fileSize=${fileSize}, isLastChunk=${isLastChunk}`);
            }

            // Read chunk from GridFS and return as base64
            const chunkData = await this.gridfsService.downloadFileRangeAsBase64(
                new ObjectId(fileId),
                bucketName,
                startByte,
                endByte
            );

            // Debug last few chunks data
            if (chunkIndex >= totalChunks - 3) {
                console.log(`🔍 DEBUG Chunk ${chunkIndex} data length: ${chunkData?.length || 0}`);
            }

            return {
                sessionId,
                chunkIndex,
                totalChunks,
                isLastChunk,
                data: chunkData // Base64 encoded chunk data
            };
        } catch (error) {
            console.error(`Error reading chunk ${chunkIndex} from GridFS:`, error);
            throw new Error(`Failed to read chunk ${chunkIndex}: ${error.message}`);
        }
    }
}