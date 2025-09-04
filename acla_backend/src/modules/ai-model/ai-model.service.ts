
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

    async getActiveModel(trackName: string, carName: string, modelType: string): Promise<AIModel> {
        const model = await this.aiModelModel
            .findOne({
                trackName,
                carName,
                modelType,
                isActive: true
            })
            .exec();

        if (!model) {
            throw new NotFoundException(
                `Active AI Model not found for track: ${trackName}, car: ${carName}, type: ${modelType}`
            );
        }
        return model;
    }

    async getActiveModelWithData(trackName: string, carName: string, modelType: string): Promise<any> {
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


    async save_imitation_learning_results(updateAiModelDto: UpdateAiModelDto): Promise<any> {
        const { trackName, carName } = updateAiModelDto;
        if (!trackName || !carName) {
            throw new InternalServerErrorException("Missing trackName or carName in results");
        }

        const existingEntry = await this.aiModelModel.findOne({
            carName: updateAiModelDto.carName,
            trackName: updateAiModelDto.trackName,
            modelType: updateAiModelDto.modelType,
            isActive: true
        });

        if (existingEntry) {
            // Update existing entry
            let modelDataFileId = existingEntry.modelDataFileId;

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
                const filename = `model_${trackName}_${carName}_${updateAiModelDto.modelType}_${Date.now()}.json`;

                modelDataFileId = await this.gridfsService.uploadJSON(
                    updateAiModelDto.modelData,
                    filename,
                    {
                        trackName,
                        carName,
                        modelType: updateAiModelDto.modelType,
                        uploadedAt: new Date()
                    },
                    GRIDFS_BUCKETS.AI_MODELS
                );
            }

            await this.aiModelModel.updateOne(
                { _id: existingEntry._id },
                {
                    $set: {
                        modelDataFileId: modelDataFileId,
                        metadata: updateAiModelDto.metadata
                    }
                }
            );
            return { updated: true };
        } else {
            // Create new entry
            const { modelData, metadata } = updateAiModelDto;
            if (!modelData) {
                throw new InternalServerErrorException("Incomplete imitation learning results");
            }

            try {
                let modelDataFileId: ObjectId | undefined;

                if (modelData) {
                    const filename = `model_${trackName}_${carName}_${updateAiModelDto.modelType}_${Date.now()}.json`;

                    modelDataFileId = await this.gridfsService.uploadJSON(
                        modelData,
                        filename,
                        {
                            trackName,
                            carName,
                            modelType: updateAiModelDto.modelType,
                            uploadedAt: new Date()
                        },
                        GRIDFS_BUCKETS.AI_MODELS
                    );
                }

                const createdEntry = new this.aiModelModel({
                    trackName,
                    carName,
                    modelType: updateAiModelDto.modelType,
                    modelDataFileId,
                    metadata,
                    isActive: true
                });
                return createdEntry.save();

            } catch (error) {
                console.error("Error saving imitation learning results:", error);
                throw new InternalServerErrorException("Failed to save imitation learning results");
            }
        }
    }

    // Additional methods for different GridFS collections

    async saveTrainingDataset(data: any, filename: string, metadata?: any): Promise<ObjectId> {
        return await this.gridfsService.uploadJSON(data, filename, metadata, GRIDFS_BUCKETS.TRAINING_DATASETS);
    }

    async getTrainingDataset(fileId: ObjectId): Promise<any> {
        return await this.gridfsService.downloadJSON(fileId, GRIDFS_BUCKETS.TRAINING_DATASETS);
    }

    async saveTelemetryData(data: any, filename: string, metadata?: any): Promise<ObjectId> {
        return await this.gridfsService.uploadJSON(data, filename, metadata, GRIDFS_BUCKETS.TELEMETRY_DATA);
    }

    async getTelemetryData(fileId: ObjectId): Promise<any> {
        return await this.gridfsService.downloadJSON(fileId, GRIDFS_BUCKETS.TELEMETRY_DATA);
    }

    async createModelBackup(modelId: string): Promise<ObjectId> {
        const model = await this.findOneWithModelData(modelId);
        const filename = `backup_${model.trackName}_${model.carName}_${model.modelType}_${Date.now()}.json`;

        return await this.gridfsService.uploadJSON(
            model,
            filename,
            {
                originalModelId: modelId,
                backupDate: new Date(),
                trackName: model.trackName,
                carName: model.carName,
                modelType: model.modelType
            },
            GRIDFS_BUCKETS.MODEL_BACKUPS
        );
    }

    async restoreModelFromBackup(backupFileId: ObjectId): Promise<AIModel> {
        const backupData = await this.gridfsService.downloadJSON(backupFileId, GRIDFS_BUCKETS.MODEL_BACKUPS);
        const { _id, createdAt, updatedAt, ...modelData } = backupData;

        return await this.create(modelData);
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