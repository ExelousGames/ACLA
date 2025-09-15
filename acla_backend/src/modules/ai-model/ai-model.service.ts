
import { Injectable, InternalServerErrorException, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { ObjectId } from 'mongodb';
import { AIModel } from '../../schemas/ai-model.schema';
import { CreateAiModelDto, UpdateAiModelDto } from './dto/ai-model.dto';
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';

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
            const filename = `model_${createAiModelDto.trackName}_${createAiModelDto.carName}_${createAiModelDto.modelType}_${Date.now()}.json`;

            modelDataFileId = await this.gridfsService.uploadJSON(
                createAiModelDto.modelData,
                filename,
                {
                    trackName: createAiModelDto.trackName,
                    carName: createAiModelDto.carName,
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
        trackName?: string;
        carName?: string;
        modelType?: string;
        isActive?: boolean;
    } = {}): Promise<AIModel[]> {
        const query: any = {};

        if (filters.trackName) query.trackName = filters.trackName;
        if (filters.carName) query.carName = filters.carName;
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
            trackName: model.trackName,
            carName: model.carName,
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
                modelWithData.modelData = await this.gridfsService.downloadJSON(model.modelDataFileId, GRIDFS_BUCKETS.AI_MODELS);
            } catch (error) {
                console.error(`Error fetching model data from GridFS: ${error.message}`);
                // Continue without model data if GridFS fetch fails
            }
        }

        return modelWithData;
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
            const filename = `model_${updateAiModelDto.trackName || existingModel.trackName}_${updateAiModelDto.carName || existingModel.carName}_${updateAiModelDto.modelType || existingModel.modelType}_${Date.now()}.json`;

            modelDataFileId = await this.gridfsService.uploadJSON(
                updateAiModelDto.modelData,
                filename,
                {
                    trackName: updateAiModelDto.trackName || existingModel.trackName,
                    carName: updateAiModelDto.carName || existingModel.carName,
                    modelType: updateAiModelDto.modelType || existingModel.modelType,
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

    // Helper to get the currently active model for a given track, car, and type
    async getActiveModel(trackName?: string, carName?: string, modelType?: string): Promise<AIModel> {
        const query: any = { isActive: true };

        if (trackName) query.trackName = trackName;
        if (carName) query.carName = carName;
        if (modelType) query.modelType = modelType;

        const model = await this.aiModelModel
            .findOne(query)
            .exec();

        if (!model) {
            const filterParts: string[] = [];
            if (trackName) filterParts.push(`track: ${trackName}`);
            if (carName) filterParts.push(`car: ${carName}`);
            if (modelType) filterParts.push(`type: ${modelType}`);

            const filterDescription = filterParts.length > 0 ? ` for ${filterParts.join(', ')}` : '';
            throw new NotFoundException(`Active AI Model not found${filterDescription}`);
        }
        return model;
    }


    /**
     * Get the currently active model for a given track, car, and type, including its model data.
     * @param trackName 
     * @param carName 
     * @param modelType 
     * @returns 
     */
    async getActiveModelWithData(trackName?: string, carName?: string, modelType?: string): Promise<any> {
        const model = await this.getActiveModel(trackName, carName, modelType);
        const modelId = (model as any)._id.toString();
        return this.findOneWithModelData(modelId);
    }

    async activateModel(id: string): Promise<AIModel> {
        const model = await this.findOne(id);

        // Deactivate all other models with same track, car, and type
        await this.aiModelModel
            .updateMany(
                {
                    trackName: model.trackName,
                    carName: model.carName,
                    modelType: model.modelType,
                    _id: { $ne: id }
                },
                { isActive: false }
            )
            .exec();

        // Activate the target model
        const activatedModel = await this.aiModelModel
            .findByIdAndUpdate(id, { isActive: true }, { new: true })
            .exec();

        return activatedModel!;
    }


    async save_ai_model(updateAiModelDto: UpdateAiModelDto): Promise<any> {
        const existingEntry = await this.findActiveModel(updateAiModelDto);

        if (existingEntry) {
            return await this.updateExistingModel(existingEntry, updateAiModelDto);
        } else {
            return await this.createNewModel(updateAiModelDto);
        }
    }

    private async findActiveModel(dto: UpdateAiModelDto): Promise<AIModel | null> {
        const query: any = { isActive: true };
        if (dto.modelType) query.modelType = dto.modelType;
        if (dto.trackName) query.trackName = dto.trackName;
        if (dto.carName) query.carName = dto.carName;

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

        try {
            const modelDataFileId = await this.uploadModelData(updateDto.modelData, updateDto);

            const createdEntry = new this.aiModelModel({
                trackName: updateDto.trackName,
                carName: updateDto.carName,
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
        const filename = `model_${dto.trackName || 'unknown'}_${dto.carName || 'unknown'}_${dto.modelType}_${Date.now()}.json`;

        return await this.gridfsService.uploadJSON(
            modelData,
            filename,
            {
                trackName: dto.trackName,
                carName: dto.carName,
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

}