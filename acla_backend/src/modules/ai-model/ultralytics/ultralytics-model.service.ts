import {
  BadRequestException,
  Injectable,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { createHash } from 'crypto';
import { createReadStream } from 'fs';
import { Model } from 'mongoose';
import { Readable } from 'stream';
import { UltralyticsModel } from '../../../schemas/ultralytics-model.schema';
import { GRIDFS_BUCKETS, GridFSService } from '../../gridfs/gridfs.service';
import {
  CreateUltralyticsModelDto,
  UltralyticsModelUpload,
} from './dto/create-ultralytics-model.dto';

@Injectable()
export class UltralyticsModelService {
  private readonly logger = new Logger(UltralyticsModelService.name);

  constructor(
    @InjectModel(UltralyticsModel.name)
    private readonly ultralyticsModels: Model<UltralyticsModel>,
    private readonly gridfsService: GridFSService,
  ) {}

  async create(dto: CreateUltralyticsModelDto, file: UltralyticsModelUpload) {
    if (!file.size) {
      throw new BadRequestException('Model file must not be empty');
    }
    const hash = createHash('sha256');
    const source = Readable.from(
      (async function* () {
        for await (const chunk of createReadStream(file.path)) {
          hash.update(chunk as Buffer);
          yield chunk;
        }
      })(),
    );
    const modelFileId = await this.gridfsService.uploadStream(
      source,
      file.originalname,
      { name: dto.name, framework: 'ultralytics', annotationFormat: 'labelme' },
      GRIDFS_BUCKETS.ULTRALYTICS_MODELS,
    );

    try {
      return await this.ultralyticsModels.create({
        name: dto.name,
        task: dto.task,
        classNames: dto.classNames,
        metadata: dto.metadata ?? {},
        framework: 'ultralytics',
        annotationFormat: 'labelme',
        modelFileId,
        filename: file.originalname,
        sizeBytes: file.size,
        sha256: hash.digest('hex'),
      });
    } catch (error) {
      await this.gridfsService
        .deleteFile(modelFileId, GRIDFS_BUCKETS.ULTRALYTICS_MODELS)
        .catch((cleanupError: unknown) => {
          this.logger.error(
            'Failed to remove unreferenced Ultralytics model file',
            cleanupError,
          );
        });
      throw error;
    }
  }

  findAll(name?: string) {
    if (name !== undefined && (typeof name !== 'string' || !name.trim())) {
      throw new BadRequestException('name must be a non-empty string');
    }
    return this.ultralyticsModels
      .find(name === undefined ? {} : { name: name.trim() })
      .sort({ createdAt: -1, _id: -1 })
      .exec();
  }

  async findOne(id: string) {
    if (!/^[a-f\d]{24}$/i.test(id)) {
      throw new BadRequestException('Invalid Ultralytics model ID');
    }
    const model = await this.ultralyticsModels.findById(id).exec();
    if (!model) {
      throw new NotFoundException('Ultralytics model not found');
    }
    return model;
  }

  async download(id: string) {
    const model = await this.findOne(id);
    const file: unknown = await this.gridfsService.getFileInfo(
      model.modelFileId,
      GRIDFS_BUCKETS.ULTRALYTICS_MODELS,
    );
    if (!file) {
      throw new NotFoundException('Ultralytics model file not found');
    }
    const stream = await this.gridfsService.downloadStream(
      model.modelFileId,
      GRIDFS_BUCKETS.ULTRALYTICS_MODELS,
    );
    return { model, stream };
  }
}
