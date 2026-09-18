import {
  Global,
  INestApplication,
  Module,
  UnauthorizedException,
} from '@nestjs/common';
import { getModelToken } from '@nestjs/mongoose';
import { AuthGuard } from '@nestjs/passport';
import { Test } from '@nestjs/testing';
import { createHash } from 'crypto';
import * as fs from 'fs/promises';
import { ObjectId } from 'mongodb';
import { Readable } from 'stream';
import * as request from 'supertest';
import { App } from 'supertest/types';
import { UltralyticsModel } from '../../../schemas/ultralytics-model.schema';
import { GRIDFS_BUCKETS, GridFSService } from '../../gridfs/gridfs.service';
import { AIModel } from '../../../schemas/ai-model.schema';
import { ChunkClientService } from '../../../shared/chunk-service/chunk.service';
import { AiModelModule } from '../ai-model.module';
import { AiModelService } from '../ai-model.service';

@Global()
@Module({
  providers: [{ provide: ChunkClientService, useValue: {} }],
  exports: [ChunkClientService],
})
class TestChunkModule {}

describe('Ultralytics model API (backend only)', () => {
  let app: INestApplication<App>;
  const id = new ObjectId().toHexString();
  const fileId = new ObjectId();
  const weights = Buffer.from([0x50, 0x4b, 0x00, 0xff, 0x80, 0x01]);
  const metadata = {
    name: 'track-segments',
    task: 'segment',
    classNames: ['straight', 'corner'],
    metadata: { epochs: 50, metrics: { map50: 0.92 } },
  };
  let storedBytes: Buffer;
  let savedModel: UltralyticsModel & { _id: string };
  const model = {
    create: jest.fn(),
    find: jest.fn(),
    findById: jest.fn(),
  };
  const legacyModelService = {
    findOne: jest.fn(),
  };
  const gridfs = {
    uploadStream: jest.fn(),
    deleteFile: jest.fn(),
    getFileInfo: jest.fn(),
    downloadStream: jest.fn(),
  };
  let unlink: jest.SpyInstance<
    ReturnType<typeof fs.unlink>,
    Parameters<typeof fs.unlink>
  >;

  beforeAll(async () => {
    const module = await Test.createTestingModule({
      imports: [TestChunkModule, AiModelModule],
    })
      .overrideProvider(getModelToken(AIModel.name))
      .useValue({})
      .overrideProvider(AiModelService)
      .useValue(legacyModelService)
      .overrideProvider(getModelToken(UltralyticsModel.name))
      .useValue(model)
      .overrideProvider(GridFSService)
      .useValue(gridfs)
      .overrideGuard(AuthGuard('jwt'))
      .useValue({
        canActivate: (context: {
          switchToHttp: () => {
            getRequest: () => { headers: Record<string, string> };
          };
        }) => {
          if (
            context.switchToHttp().getRequest().headers.authorization !==
            'Bearer backend-test'
          ) {
            throw new UnauthorizedException();
          }
          return true;
        },
      })
      .compile();
    app = module.createNestApplication();
    app.useLogger(false);
    await app.init();
  });

  beforeEach(() => {
    jest.clearAllMocks();
    unlink = jest.spyOn(fs, 'unlink');
    model.create.mockImplementation((data: UltralyticsModel) => {
      savedModel = { ...data, _id: id };
      return Promise.resolve(savedModel);
    });
    gridfs.uploadStream.mockImplementation(async (stream: Readable) => {
      const chunks: Buffer[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk as Buffer);
      }
      storedBytes = Buffer.concat(chunks);
      return fileId;
    });
    gridfs.deleteFile.mockResolvedValue(undefined);
    model.findById.mockImplementation(() => ({
      exec: () => Promise.resolve(savedModel),
    }));
    gridfs.getFileInfo.mockResolvedValue({ length: weights.length });
    gridfs.downloadStream.mockImplementation(() =>
      Promise.resolve(Readable.from([storedBytes])),
    );
  });

  afterEach(() => jest.restoreAllMocks());
  afterAll(async () => {
    await app?.close();
  });

  const upload = (
    raw = JSON.stringify(metadata),
    bytes = weights,
    filename = 'best.pt',
  ) =>
    request(app.getHttpServer())
      .post('/ai-model/ultralytics')
      .set('Authorization', 'Bearer backend-test')
      .field('metadata', raw)
      .attach('file', bytes, filename);

  const expectTemporaryFileRemoved = async () => {
    expect(unlink).toHaveBeenCalledTimes(1);
    const temporaryPath = unlink.mock.calls[0][0] as string;
    await expect(fs.stat(temporaryPath)).rejects.toMatchObject({
      code: 'ENOENT',
    });
  };

  it('stores exact binary weights and ordered labels, then streams the original file', async () => {
    const response = await upload().expect(201);
    expect(response.body).toMatchObject({
      ...metadata,
      _id: id,
      framework: 'ultralytics',
      annotationFormat: 'labelme',
      filename: 'best.pt',
      sizeBytes: weights.length,
      modelFileId: fileId.toHexString(),
      sha256: createHash('sha256').update(weights).digest('hex'),
    });
    expect(storedBytes).toEqual(weights);
    expect(gridfs.uploadStream).toHaveBeenCalledWith(
      expect.any(Readable),
      'best.pt',
      expect.objectContaining({ name: metadata.name }),
      GRIDFS_BUCKETS.ULTRALYTICS_MODELS,
    );
    await expectTemporaryFileRemoved();

    const download = await request(app.getHttpServer())
      .get(`/ai-model/ultralytics/${id}/file`)
      .set('Authorization', 'Bearer backend-test')
      .expect(200)
      .expect('Content-Type', 'application/octet-stream')
      .expect('Content-Length', String(weights.length));
    expect(download.body).toEqual(weights);
    expect(download.headers['content-disposition']).toContain('best.pt');
    expect(gridfs.downloadStream).toHaveBeenCalledWith(
      fileId,
      GRIDFS_BUCKETS.ULTRALYTICS_MODELS,
    );
  });

  it('requires authentication before accepting files', async () => {
    await request(app.getHttpServer())
      .post('/ai-model/ultralytics')
      .field('metadata', JSON.stringify(metadata))
      .attach('file', weights, 'best.pt')
      .expect(401);
    expect(gridfs.uploadStream).not.toHaveBeenCalled();
    expect(unlink).not.toHaveBeenCalled();
  });

  it('requires authentication for the Ultralytics list instead of treating the family as a generic ID', async () => {
    await request(app.getHttpServer()).get('/ai-model/ultralytics').expect(401);
    expect(legacyModelService.findOne).not.toHaveBeenCalled();
    expect(model.find).not.toHaveBeenCalled();
  });

  it('keeps the existing generic model lookup available', async () => {
    const existingModel = { _id: id, modelType: 'imitation_learning' };
    legacyModelService.findOne.mockResolvedValueOnce(existingModel);
    await request(app.getHttpServer())
      .get(`/ai-model/${id}`)
      .expect(200, existingModel);
    expect(legacyModelService.findOne).toHaveBeenCalledWith(id);
    expect(model.findById).not.toHaveBeenCalled();
  });

  it.each([
    'invalid JSON',
    'null',
    '[]',
    JSON.stringify({ ...metadata, name: '' }),
    JSON.stringify({ ...metadata, task: 'unknown' }),
    JSON.stringify({ ...metadata, classNames: [] }),
    JSON.stringify({ ...metadata, classNames: ['corner', 'corner'] }),
    JSON.stringify({ ...metadata, classNames: [1] }),
    JSON.stringify({ ...metadata, metadata: [] }),
  ])(
    'rejects invalid metadata and removes the temporary file: %s',
    async (raw) => {
      await upload(raw).expect(400);
      expect(gridfs.uploadStream).not.toHaveBeenCalled();
      expect(model.create).not.toHaveBeenCalled();
      await expectTemporaryFileRemoved();
    },
  );

  it('rejects a missing file', async () => {
    await request(app.getHttpServer())
      .post('/ai-model/ultralytics')
      .set('Authorization', 'Bearer backend-test')
      .field('metadata', JSON.stringify(metadata))
      .expect(400);
    expect(model.create).not.toHaveBeenCalled();
  });

  it('rejects missing metadata and cleans up the uploaded file', async () => {
    await request(app.getHttpServer())
      .post('/ai-model/ultralytics')
      .set('Authorization', 'Bearer backend-test')
      .attach('file', weights, 'best.pt')
      .expect(400);
    expect(gridfs.uploadStream).not.toHaveBeenCalled();
    await expectTemporaryFileRemoved();
  });

  it('rejects empty files before writing to GridFS', async () => {
    await upload(JSON.stringify(metadata), Buffer.alloc(0)).expect(400);
    expect(gridfs.uploadStream).not.toHaveBeenCalled();
    await expectTemporaryFileRemoved();
  });

  it('rejects unsupported file extensions', async () => {
    await upload(JSON.stringify(metadata), weights, 'labels.json').expect(400);
    expect(gridfs.uploadStream).not.toHaveBeenCalled();
  });

  it('deletes GridFS weights when saving the metadata fails', async () => {
    model.create.mockRejectedValueOnce(new Error('Database write failed'));
    await upload().expect(500);
    expect(gridfs.deleteFile).toHaveBeenCalledWith(
      fileId,
      GRIDFS_BUCKETS.ULTRALYTICS_MODELS,
    );
    await expectTemporaryFileRemoved();
  });

  it('does not create metadata when GridFS fails and still removes the temporary file', async () => {
    gridfs.uploadStream.mockRejectedValueOnce(new Error('GridFS unavailable'));
    await upload().expect(500);
    expect(model.create).not.toHaveBeenCalled();
    await expectTemporaryFileRemoved();
  });

  it('lists stored metadata with an exact name filter, newest first', async () => {
    const sort = jest.fn().mockReturnValue({ exec: () => Promise.resolve([]) });
    model.find.mockReturnValue({ sort });
    await request(app.getHttpServer())
      .get('/ai-model/ultralytics?name=track-segments')
      .set('Authorization', 'Bearer backend-test')
      .expect(200, []);
    expect(model.find).toHaveBeenCalledWith({ name: 'track-segments' });
    expect(sort).toHaveBeenCalledWith({ createdAt: -1, _id: -1 });
  });

  it('rejects malformed IDs without querying MongoDB', async () => {
    await request(app.getHttpServer())
      .get('/ai-model/ultralytics/invalid')
      .set('Authorization', 'Bearer backend-test')
      .expect(400);
    expect(model.findById).not.toHaveBeenCalled();
  });

  it('returns 404 for a missing model', async () => {
    model.findById.mockReturnValueOnce({ exec: () => Promise.resolve(null) });
    await request(app.getHttpServer())
      .get(`/ai-model/ultralytics/${id}`)
      .set('Authorization', 'Bearer backend-test')
      .expect(404);
  });

  it('returns 404 when the stored file is missing', async () => {
    await upload().expect(201);
    gridfs.getFileInfo.mockResolvedValueOnce(null);
    await request(app.getHttpServer())
      .get(`/ai-model/ultralytics/${id}/file`)
      .set('Authorization', 'Bearer backend-test')
      .expect(404);
    expect(gridfs.downloadStream).not.toHaveBeenCalled();
  });
});
