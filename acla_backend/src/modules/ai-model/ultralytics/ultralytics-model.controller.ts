import {
  BadRequestException,
  Body,
  Controller,
  Get,
  Header,
  Logger,
  Param,
  Post,
  Query,
  StreamableFile,
  UploadedFile,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { FileInterceptor } from '@nestjs/platform-express';
import { unlink } from 'fs/promises';
import { tmpdir } from 'os';
import { extname, join } from 'path';
import {
  parseUltralyticsModelMetadata,
  UltralyticsModelUpload,
} from './dto/create-ultralytics-model.dto';
import { UltralyticsModelService } from './ultralytics-model.service';

@UseGuards(AuthGuard('jwt'))
@Controller('ai-model/ultralytics')
export class UltralyticsModelController {
  private readonly logger = new Logger(UltralyticsModelController.name);

  constructor(
    private readonly ultralyticsModelService: UltralyticsModelService,
  ) {}

  @Post()
  @UseInterceptors(
    FileInterceptor('file', {
      dest: join(tmpdir(), 'acla-ultralytics-models'),
      limits: {
        fileSize: 512 * 1024 * 1024,
        files: 1,
        fields: 1,
        fieldSize: 1024 * 1024,
      },
      fileFilter: (_req, file, callback) => {
        if (extname(file.originalname).toLowerCase() !== '.pt') {
          return callback(
            new BadRequestException('Model file must use the .pt extension'),
            false,
          );
        }
        callback(null, true);
      },
    }),
  )
  async create(
    @UploadedFile() file: UltralyticsModelUpload | undefined,
    @Body('metadata') metadata: unknown,
  ) {
    if (!file) {
      throw new BadRequestException(
        'A model file is required in the file field',
      );
    }
    try {
      return await this.ultralyticsModelService.create(
        parseUltralyticsModelMetadata(metadata),
        file,
      );
    } finally {
      await unlink(file.path).catch((error: unknown) => {
        this.logger.warn(
          `Failed to remove temporary Ultralytics model upload: ${String(error)}`,
        );
      });
    }
  }

  @Get()
  findAll(@Query('name') name?: string) {
    return this.ultralyticsModelService.findAll(name);
  }

  @Get('track-vision')
  @Header('Cache-Control', 'no-store')
  trackVision() {
    return this.ultralyticsModelService.findTrackVisionModel();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.ultralyticsModelService.findOne(id);
  }

  @Get(':id/file')
  async download(@Param('id') id: string) {
    const { model, stream } = await this.ultralyticsModelService.download(id);
    return new StreamableFile(stream, {
      type: 'application/octet-stream',
      disposition: `attachment; filename="model.pt"; filename*=UTF-8''${encodeURIComponent(model.filename)}`,
      length: model.sizeBytes,
    });
  }
}
