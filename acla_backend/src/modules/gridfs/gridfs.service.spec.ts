import { ObjectId } from 'mongodb';
import { Connection } from 'mongoose';
import { Readable, Writable } from 'stream';
import { GridFSService, GRIDFS_BUCKETS } from './gridfs.service';

describe('GridFS stream uploads', () => {
  const fileId = new ObjectId();
  let service: GridFSService;
  let upload: Writable & { id: ObjectId; abort: jest.Mock };
  let received: Buffer[];
  let deleteFile: jest.Mock;

  beforeEach(() => {
    received = [];
    deleteFile = jest.fn().mockResolvedValue(undefined);
    upload = Object.assign(
      new Writable({
        write(chunk: Buffer, _encoding, callback) {
          received.push(chunk);
          callback();
        },
      }),
      { id: fileId, abort: jest.fn().mockResolvedValue(undefined) },
    );
    service = new GridFSService({} as Connection);
    jest
      .spyOn(
        service as unknown as { getBucket: () => Promise<unknown> },
        'getBucket',
      )
      .mockResolvedValue({
        openUploadStream: () => upload,
        delete: deleteFile,
      });
  });

  afterEach(() => jest.restoreAllMocks());

  it('preserves binary bytes', async () => {
    const bytes = Buffer.from([0, 255, 128, 1]);
    await expect(
      service.uploadStream(
        Readable.from([bytes]),
        'best.pt',
        {},
        GRIDFS_BUCKETS.ULTRALYTICS_MODELS,
      ),
    ).resolves.toEqual(fileId);
    expect(Buffer.concat(received)).toEqual(bytes);
    expect(upload.abort).not.toHaveBeenCalled();
  });

  it('aborts partial GridFS uploads when the input stream fails', async () => {
    const source = Readable.from(
      (async function* () {
        yield Buffer.from('partial');
        await Promise.resolve();
        throw new Error('Source read failed');
      })(),
    );
    await expect(service.uploadStream(source, 'best.pt')).rejects.toThrow(
      'Source read failed',
    );
    expect(upload.abort).toHaveBeenCalledTimes(1);
    expect(source.destroyed).toBe(true);
    expect(upload.destroyed).toBe(true);
  });

  it('aborts and destroys both streams when the upload times out', async () => {
    const controller = new AbortController();
    jest.spyOn(AbortSignal, 'timeout').mockReturnValue(controller.signal);
    const source = new Readable({
      read() {
        controller.abort();
      },
    });
    await expect(service.uploadStream(source, 'best.pt')).rejects.toThrow();
    expect(upload.abort).toHaveBeenCalledTimes(1);
    expect(source.destroyed).toBe(true);
    expect(upload.destroyed).toBe(true);
  });
  it('deletes partial files when GridFS finalization fails and abort is no longer allowed', async () => {
    upload._final = (callback) => callback(new Error('Final write failed'));
    upload.abort.mockRejectedValueOnce(
      new Error('Cannot abort a stream that has already completed'),
    );
    await expect(
      service.uploadStream(Readable.from([Buffer.from('weights')]), 'best.pt'),
    ).rejects.toThrow('Final write failed');
    expect(deleteFile).toHaveBeenCalledWith(fileId);
  });
});
