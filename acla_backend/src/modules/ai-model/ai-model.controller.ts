

import { Controller, Get, Post, Put, Delete, Body, Param, Query } from '@nestjs/common';
import { AiModelService } from './ai-model.service';
import { CreateAiModelDto, UpdateAiModelDto } from './dto/ai-model.dto';
import { ChunkClientService } from '../../shared/chunk-service/chunk.service';
import { ChunkData } from '../../shared/chunk-service/interfaces/chunk.interface';
import { GridFSService } from '../gridfs/gridfs.service';

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
        return this.aiModelService.findOneWithModelData(id);
    }

    @Put(':id')
    async update(@Param('id') id: string, @Body() updateAiModelDto: UpdateAiModelDto) {
        return this.aiModelService.update(id, updateAiModelDto);
    }

    @Delete(':id')
    async remove(@Param('id') id: string) {
        return this.aiModelService.remove(id);
    }

    @Get('active/:trackName/:carName/:modelType')
    async getActiveModel(
        @Param('trackName') trackName: string,
        @Param('carName') carName: string,
        @Param('modelType') modelType: string
    ) {
        return this.aiModelService.getActiveModel(trackName, carName, modelType);
    }

    @Get('active/:trackName/:carName/:modelType/with-data')
    async getActiveModelWithData(
        @Param('trackName') trackName: string,
        @Param('carName') carName: string,
        @Param('modelType') modelType: string
    ) {
        return this.aiModelService.getActiveModelWithData(trackName, carName, modelType);
    }

    @Post(':id/activate')
    async activateModel(@Param('id') id: string) {
        return this.aiModelService.activateModel(id);
    }

    @Post('imitation-learning/save')
    async save_imitation_learning_results(@Body() chunkData: ChunkData) {
        console.log("Received chunked request to save imitation learning results:", {
            sessionId: chunkData.sessionId,
            chunkIndex: chunkData.chunkIndex,
            totalChunks: chunkData.totalChunks
        });

        return this.chunkService.handleIncomingChunk(
            chunkData,
            async (completeData: UpdateAiModelDto) => {
                console.log("Processing complete imitation learning data");
                return this.aiModelService.save_imitation_learning_results(completeData);
            }
        );
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

    @Post(':id/backup')
    async createBackup(@Param('id') id: string) {
        const backupFileId = await this.aiModelService.createModelBackup(id);
        return {
            message: 'Backup created successfully',
            backupFileId: backupFileId.toString()
        };
    }

    @Post('restore/:backupFileId')
    async restoreFromBackup(@Param('backupFileId') backupFileId: string) {
        const restoredModel = await this.aiModelService.restoreModelFromBackup(new (require('mongodb').ObjectId)(backupFileId));
        return {
            message: 'Model restored successfully',
            model: restoredModel
        };
    }
}