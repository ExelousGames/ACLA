

import { Controller, Get, Post, Put, Delete, Body, Param, Query } from '@nestjs/common';
import { AiModelService } from './ai-model.service';
import { CreateAiModelDto, UpdateAiModelDto } from './dto/ai-model.dto';
import { ChunkClientService } from '../../shared/chunk-service/chunk.service';
import { ChunkData } from '../../shared/chunk-service/interfaces/chunk.interface';
import { GridFSService } from '../gridfs/gridfs.service';
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
     * Prepare the active model data for chunked transfer (returns session info only).
     * This is the recommended way to handle large model data.
     * Use this endpoint followed by individual calls to /chunked-data/:sessionId/:chunkIndex
     * @param modelType Required - The type of model to retrieve
     * @param trackName Optional - The track name to filter by (query parameter)
     * @param carName Optional - The car name to filter by (query parameter)
     * @returns Session information for chunked transfer
     */
    @Get('active/:modelType/prepare-chunked')
    async initGetActiveModelData(
        @Param('modelType') modelType: string,
        @Query('trackName') trackName?: string,
        @Query('carName') carName?: string
    ) {
        console.log(`Preparing chunked data for active model - Track: ${trackName}, Car: ${carName}, Type: ${modelType}`);

        try {
            const modelData = await this.aiModelService.getActiveModelWithData(trackName, carName, modelType);

            // Check if the model data contains an error or is too large
            if (modelData.modelDataInfo?.isLargeFile) {
                return {
                    success: false,
                    message: 'Model data is too large to process. This indicates a system limitation with the current model size.',
                    fileSize: modelData.modelDataInfo.fileSizeHuman,
                    recommendation: 'Consider reducing the model size or implementing streaming JSON processing for this specific model type.'
                };
            }

            if (modelData.modelDataError) {
                return {
                    success: false,
                    message: `Failed to prepare model data: ${modelData.modelDataError.message}`,
                    timestamp: modelData.modelDataError.timestamp,
                    recommendation: modelData.modelDataError.recommendedAction
                };
            }

            const result = await this.chunkService.prepareDataForChunkedSending(modelData);

            // Return only session info, not the actual chunks
            return {
                success: result.success,
                sessionId: result.sessionId,
                totalChunks: result.totalChunks,
                message: result.message
            };
        } catch (error) {
            console.error(`Error preparing chunked data: ${error.message}`);
            return {
                success: false,
                message: `Failed to prepare chunked data: ${error.message}`,
                recommendation: 'Check server logs for detailed error information. If this is a very large model, consider using alternative data handling approaches.'
            };
        }
    }

    /**
     * Get a specific chunk from a prepared chunked session.
     * @param sessionId The session ID from prepare-chunked endpoint
     * @param chunkIndex The index of the chunk to retrieve (0-based)
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
            // Get the session status to validate it exists
            const sessionStatus = await this.chunkService.getSessionStatus(sessionId);

            if (!sessionStatus.success) {
                return {
                    success: false,
                    message: 'Session not found or expired'
                };
            }

            // Get the chunk directly from the chunk service
            const result = await this.chunkService.getPreparedChunk(sessionId, chunkIndexNum);
            return result;
        } catch (error) {
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
}