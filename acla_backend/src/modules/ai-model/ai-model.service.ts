
import { Injectable, InternalServerErrorException, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { ObjectId } from 'mongodb';
import { AIModel } from '../../schemas/ai-model.schema';
import { CreateAiModelDto, UpdateAiModelDto } from './dto/ai-model.dto';
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { Readable } from 'stream';

@Injectable()
export class AiModelService {
    constructor(
        @InjectModel(AIModel.name) private aiModelModel: Model<AIModel>,
        private readonly gridfsService: GridFSService,
    ) { }

    async create(createAiModelDto: CreateAiModelDto): Promise<AIModel> {
        let modelDataFileId: ObjectId | undefined;

        // If modelData is provided, upload it to GridFS
        if (createAiModelDto.modelData) {
            const filename = this.buildModelFilename(createAiModelDto.modelType);

            modelDataFileId = await this.gridfsService.uploadJSON(
                createAiModelDto.modelData,
                filename,
                {
                    modelType: createAiModelDto.modelType,
                    uploadedAt: new Date()
                },
                GRIDFS_BUCKETS.AI_MODELS
            );
        }

        //seperate modelData from the rest of the DTO  
        const { modelData, ...modelDataWithoutFile } = createAiModelDto;
        const createdModel = new this.aiModelModel({
            ...modelDataWithoutFile,
            modelDataFileId
        });

        return createdModel.save();
    }

    async findAll(filters: {
        modelType?: string;
        isActive?: boolean;
    } = {}): Promise<AIModel[]> {
        const query: any = {};

        if (filters.modelType) query.modelType = filters.modelType;
        if (filters.isActive !== undefined) query.isActive = filters.isActive;

        return this.aiModelModel.find(query).exec();
    }

    async findOne(id: string): Promise<AIModel> {
        const model = await this.aiModelModel.findById(id).exec();
        if (!model) {
            throw new NotFoundException(`AI Model with ID ${id} not found`);
        }
        return model;
    }

    /**
     * Find an AI model by its ID and include the model data retrieved from GridFS.
     * @param id Model ID
     * @returns AI Model with its data
     */
    async findOneWithModelData(id: string): Promise<any> {
        const model = await this.findOne(id);
        const modelWithData: any = {
            _id: (model as any)._id,
            modelType: model.modelType,
            modelDataFileId: model.modelDataFileId,
            metadata: model.metadata,
            isActive: model.isActive,
            createdAt: (model as any).createdAt,
            updatedAt: (model as any).updatedAt
        };

        // Fetch model data from GridFS if available
        if (model.modelDataFileId) {
            try {
                // Check file size first to determine if we need to use streaming
                const fileSize = await this.gridfsService.getFileSize(model.modelDataFileId, GRIDFS_BUCKETS.AI_MODELS);
                const maxSafeSize = 50 * 1024 * 1024; // 50MB - much more conservative limit

                if (fileSize >= maxSafeSize) {
                    // Instead of loading the data, return metadata about the file
                    modelWithData.modelDataInfo = {
                        fileSize: fileSize,
                        fileSizeHuman: this.formatFileSize(fileSize),
                        isLargeFile: true,
                        message: 'Model data is too large to return directly. Use chunked data endpoints.',
                        recommendedAction: 'Use /active/:modelType/prepare-chunked followed by /active/chunked-data/:sessionId/:chunkIndex endpoints'
                    };
                } else {
                    // File is small enough to load normally
                    modelWithData.modelData = await this.gridfsService.downloadJSON(model.modelDataFileId, GRIDFS_BUCKETS.AI_MODELS);
                }
            } catch (error) {
                console.error(`Error fetching model data from GridFS: ${error.message}`);
                modelWithData.modelDataError = {
                    message: `Failed to load model data: ${error.message}`,
                    timestamp: new Date().toISOString(),
                    recommendedAction: 'Use chunked data endpoints for large files'
                };
            }
        }

        return modelWithData;
    }

    /**
     * Helper method to format file size in human-readable format
     */
    /**
     * Helper method to format file size in human-readable format
     */
    private formatFileSize(bytes: number): string {
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    private buildModelFilename(modelType?: string): string {
        const safeModelType = modelType || 'generic';
        return `model_${safeModelType}_${Date.now()}.json`;
    }

    /**
     * Get a specific chunk from a prepared session
     */
    async getChunkFromSession(sessionId: string, chunkIndex: number): Promise<any> {
        // For now, we'll delegate to the chunk service
        // In a more advanced implementation, you might have additional model-specific logic here
        throw new Error('This method should not be called directly. Use the chunk service instead.');
    }

    async update(id: string, updateAiModelDto: UpdateAiModelDto): Promise<AIModel> {
        const existingModel = await this.findOne(id);
        let modelDataFileId = existingModel.modelDataFileId;

        // If new model data is provided, upload it to GridFS
        if (updateAiModelDto.modelData) {
            // Delete old file if it exists
            if (modelDataFileId) {
                try {
                    await this.gridfsService.deleteFile(modelDataFileId, GRIDFS_BUCKETS.AI_MODELS);
                } catch (error) {
                    console.error(`Error deleting old model data from GridFS: ${error.message}`);
                }
            }

            // Upload new model data
            const nextModelType = updateAiModelDto.modelType || existingModel.modelType;
            const filename = this.buildModelFilename(nextModelType);

            modelDataFileId = await this.gridfsService.uploadJSON(
                updateAiModelDto.modelData,
                filename,
                {
                    modelType: nextModelType,
                    uploadedAt: new Date()
                },
                GRIDFS_BUCKETS.AI_MODELS
            );
        }

        const { modelData, ...updateDataWithoutFile } = updateAiModelDto;
        const updatedModel = await this.aiModelModel
            .findByIdAndUpdate(
                id,
                { ...updateDataWithoutFile, modelDataFileId },
                { new: true }
            )
            .exec();

        if (!updatedModel) {
            throw new NotFoundException(`AI Model with ID ${id} not found`);
        }
        return updatedModel;
    }

    async remove(id: string): Promise<void> {
        const model = await this.findOne(id);

        // Delete associated GridFS file if it exists
        if (model.modelDataFileId) {
            try {
                await this.gridfsService.deleteFile(model.modelDataFileId, GRIDFS_BUCKETS.AI_MODELS);
            } catch (error) {
                console.error(`Error deleting model data from GridFS: ${error.message}`);
                // Continue with model deletion even if GridFS deletion fails
            }
        }

        const result = await this.aiModelModel.findByIdAndDelete(id).exec();
        if (!result) {
            throw new NotFoundException(`AI Model with ID ${id} not found`);
        }
    }

    // Helper to get the currently active model for a given type
    async getActiveModel(modelType?: string): Promise<AIModel> {
        const query: any = { isActive: true };

        if (modelType) query.modelType = modelType;

        const model = await this.aiModelModel
            .findOne(query)
            .exec();

        if (!model) {
            const filterDescription = modelType ? ` for type: ${modelType}` : '';
            throw new NotFoundException(`Active AI Model not found${filterDescription}`);
        }
        return model;
    }


    /**
     * Get the currently active model for a given type, including its model data.
     * @param modelType 
     * @returns 
     */
    async getActiveModelWithData(modelType?: string): Promise<any> {
        const model = await this.getActiveModel(modelType);
        const modelId = (model as any)._id.toString();
        return this.findOneWithModelData(modelId);
    }

    async activateModel(id: string): Promise<AIModel> {
        const model = await this.findOne(id);

        await this.aiModelModel
            .updateMany(
                {
                    modelType: model.modelType,
                    _id: { $ne: id }
                },
                { isActive: false }
            )
            .exec();

        const activatedModel = await this.aiModelModel
            .findByIdAndUpdate(id, { isActive: true }, { new: true })
            .exec();

        if (!activatedModel) {
            throw new NotFoundException(`AI Model with ID ${id} not found`);
        }

        return activatedModel;
    }

    async save_ai_model(updateAiModelDto: UpdateAiModelDto): Promise<any> {
        try {
            const existingEntry = await this.findActiveModel(updateAiModelDto.modelType);

            if (existingEntry) {
                return await this.updateExistingModel(existingEntry, updateAiModelDto);
            } else {
                return await this.createNewModel(updateAiModelDto);
            }
        } catch (error) {
            console.error("Error in save_ai_model:", error);
            throw error;
        }
    }

    async save_ai_model_from_file(filePath: string): Promise<any> {
        const stat = await fs.promises.stat(filePath);
        const isHuge = stat.size > 100 * 1024 * 1024; // >100MB

        if (!isHuge) {
            const fileContent = await fs.promises.readFile(filePath, 'utf8');
            const dto = JSON.parse(fileContent) as UpdateAiModelDto;
            return this.save_ai_model(dto);
        }

        let modelType = 'unknown';
        let metadata: any = undefined;

        try {
            const extractedData = await this.extractModelDataFromHugeFile(filePath);

            modelType = extractedData.modelType || modelType;
            metadata = extractedData.metadata;

            if (!extractedData.modelDataFilePath) {
                throw new Error('Failed to extract modelData from huge file');
            }

            const filename = this.buildModelFilename(modelType);
            const modelDataStream = fs.createReadStream(extractedData.modelDataFilePath);
            const fileId = await this.gridfsService.uploadStream(
                modelDataStream as unknown as Readable,
                filename,
                { modelType, uploadedAt: new Date() },
                GRIDFS_BUCKETS.AI_MODELS
            );

            try {
                await fs.promises.unlink(extractedData.modelDataFilePath);
            } catch (cleanupError) {
                console.warn(`Failed to cleanup temp file ${extractedData.modelDataFilePath}: ${cleanupError.message}`);
            }

            const existing = await this.findActiveModel(modelType);
            if (existing) {
                await this.aiModelModel.updateOne(
                    { _id: (existing as any)._id },
                    { $set: { modelDataFileId: fileId, metadata } }
                );
                return { updated: true, fileId };
            }

            const created = new this.aiModelModel({
                modelType,
                modelDataFileId: fileId,
                metadata,
                isActive: true
            });
            return await created.save();

        } catch (error) {
            console.error(`Error extracting modelData from huge file: ${error.message}`);
            throw new InternalServerErrorException(`Failed to extract modelData from file: ${error.message}`);
        }
    }

    private async findActiveModel(modelType?: string): Promise<AIModel | null> {
        const query: any = { isActive: true };
        if (modelType) query.modelType = modelType;

        return await this.aiModelModel.findOne(query);
    }

    private async updateExistingModel(existingEntry: AIModel, updateDto: UpdateAiModelDto): Promise<any> {
        let modelDataFileId = existingEntry.modelDataFileId;

        if (updateDto.modelData) {
            modelDataFileId = await this.replaceModelData(
                existingEntry.modelDataFileId,
                updateDto.modelData,
                updateDto
            );
        }

        await this.aiModelModel.updateOne(
            { _id: (existingEntry as any)._id },
            {
                $set: {
                    modelDataFileId: modelDataFileId,
                    metadata: updateDto.metadata
                }
            }
        );

        return { updated: true };
    }

    private async createNewModel(updateDto: UpdateAiModelDto): Promise<AIModel> {
        if (!updateDto.modelData) {
            throw new InternalServerErrorException("Incomplete imitation learning results");
        }

        if (!updateDto.modelType) {
            throw new InternalServerErrorException("Model type is required to create a new AI model");
        }

        try {
            const modelDataFileId = await this.uploadModelData(updateDto.modelData, updateDto);

            const createdEntry = new this.aiModelModel({
                modelType: updateDto.modelType,
                modelDataFileId,
                metadata: updateDto.metadata,
                isActive: true
            });

            return createdEntry.save();
        } catch (error) {
            console.error("Error saving imitation learning results:", error);
            throw new InternalServerErrorException("Failed to save imitation learning results");
        }
    }

    private async replaceModelData(
        oldFileId: ObjectId | undefined,
        newModelData: any,
        dto: UpdateAiModelDto
    ): Promise<ObjectId> {
        // Delete old file if it exists
        if (oldFileId) {
            try {
                await this.gridfsService.deleteFile(oldFileId, GRIDFS_BUCKETS.AI_MODELS);
            } catch (error) {
                console.error(`Error deleting old model data from GridFS: ${error.message}`);
            }
        }

        return await this.uploadModelData(newModelData, dto);
    }

    private async uploadModelData(modelData: any, dto: UpdateAiModelDto): Promise<ObjectId> {
        const filename = this.buildModelFilename(dto.modelType);

        return await this.gridfsService.uploadJSON(
            modelData,
            filename,
            {
                modelType: dto.modelType,
                uploadedAt: new Date()
            },
            GRIDFS_BUCKETS.AI_MODELS
        );
    }

    // Get GridFS statistics for different buckets
    async getGridFSStats(): Promise<any> {
        return await this.gridfsService.getAllBucketsStats();
    }

    // Health check for GridFS
    async isGridFSHealthy(): Promise<boolean> {
        return await this.gridfsService.isHealthy(GRIDFS_BUCKETS.AI_MODELS);
    }

    // Get GridFS service info
    async getGridFSServiceInfo(): Promise<any> {
        return await this.gridfsService.getServiceInfo();
    }

    // Reinitialize GridFS
    async reinitializeGridFS(): Promise<void> {
        return await this.gridfsService.reinitialize();
    }

    /**
     * Extract modelData from huge files using streaming approach to avoid memory issues.
     * This method streams through the file to find and extract only the modelData portion.
     */
    private async extractModelDataFromHugeFile(filePath: string): Promise<{
        modelType?: string,
        metadata?: any,
        modelDataFilePath?: string
    }> {
        return new Promise((resolve, reject) => {

            let buffer = '';
            let foundModelDataStart = false;
            let braceCount = 0;
            let insideModelData = false;
            let modelType = 'unknown';
            let metadata: any = undefined;

            // Create temporary file for extracted modelData
            const tempDir = os.tmpdir();
            const tempFilename = `modeldata_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.json`;
            const tempFilePath = path.join(tempDir, tempFilename);
            const writeStream = fs.createWriteStream(tempFilePath);

            const readStream = fs.createReadStream(filePath, { encoding: 'utf8', highWaterMark: 64 * 1024 });

            readStream.on('data', (chunk: string) => {
                buffer += chunk;

                if (modelType === 'unknown') {
                    const mtMatch = buffer.match(/"modelType"\s*:\s*"([^"]+)"/);
                    if (mtMatch) modelType = mtMatch[1];
                }

                // Look for the start of modelData
                if (!foundModelDataStart) {
                    const modelDataStartIndex = buffer.indexOf('"modelData":');
                    if (modelDataStartIndex !== -1) {
                        foundModelDataStart = true;
                        insideModelData = true;

                        // Find the opening brace/bracket after "modelData":
                        const valueStartIndex = buffer.indexOf(':', modelDataStartIndex) + 1;
                        let valueStart = valueStartIndex;
                        while (valueStart < buffer.length && /\s/.test(buffer[valueStart])) {
                            valueStart++;
                        }

                        if (valueStart < buffer.length) {
                            const firstChar = buffer[valueStart];
                            braceCount = firstChar === '{' ? 1 : (firstChar === '[' ? 1 : 0);

                            // Process the modelData content starting from the opening brace/bracket
                            const modelDataStart = buffer.substring(valueStart);

                            // Find if the modelData ends in this chunk
                            let endIndex = -1;
                            for (let i = 1; i < modelDataStart.length; i++) {
                                if (modelDataStart[i] === '{' || modelDataStart[i] === '[') {
                                    braceCount++;
                                } else if (modelDataStart[i] === '}' || modelDataStart[i] === ']') {
                                    braceCount--;
                                    if (braceCount === 0) {
                                        // Found the end of modelData in this chunk
                                        endIndex = i + 1; // Include the closing brace
                                        break;
                                    }
                                }
                            }

                            if (endIndex !== -1) {
                                // Write only up to the end of modelData
                                writeStream.write(modelDataStart.substring(0, endIndex));
                                writeStream.end();
                                readStream.destroy();
                                return;
                            } else {
                                // Haven't found the end yet, write the entire chunk
                                writeStream.write(modelDataStart);
                            }
                        }
                        // Clear buffer after processing
                        buffer = '';
                    }
                } else if (insideModelData) {
                    // We're inside modelData, find if it ends in this chunk
                    let endIndex = -1;

                    for (let i = 0; i < buffer.length; i++) {
                        if (buffer[i] === '{' || buffer[i] === '[') {
                            braceCount++;
                        } else if (buffer[i] === '}' || buffer[i] === ']') {
                            braceCount--;
                            if (braceCount === 0) {
                                // Found the end of modelData
                                endIndex = i + 1; // Include the closing brace
                                break;
                            }
                        }
                    }

                    if (endIndex !== -1) {
                        // Write only up to the end of modelData
                        writeStream.write(buffer.substring(0, endIndex));
                        writeStream.end();
                        readStream.destroy();
                        return;
                    } else {
                        // Haven't found the end yet, write the entire chunk
                        writeStream.write(buffer);
                    }

                    buffer = '';
                }

                // Keep buffer size manageable when not inside modelData
                if (!insideModelData && buffer.length > 1024 * 1024) { // 1MB
                    buffer = buffer.substring(buffer.length - 512 * 1024); // Keep last 512KB
                }
            });

            readStream.on('end', () => {
                if (insideModelData) {
                    // If we reached the end and we're still inside modelData, end the write stream
                    writeStream.end();
                } else if (!foundModelDataStart) {
                    // No modelData found in the file
                    writeStream.destroy();
                    reject(new Error('No modelData field found in the file'));
                }
            });

            writeStream.on('finish', () => {
                resolve({
                    modelType,
                    metadata,
                    modelDataFilePath: tempFilePath
                });
            });

            readStream.on('error', (error) => {
                writeStream.destroy();
                reject(new Error(`Error reading file: ${error.message}`));
            });

            writeStream.on('error', (error) => {
                reject(new Error(`Error writing temp file: ${error.message}`));
            });
        });
    }

}