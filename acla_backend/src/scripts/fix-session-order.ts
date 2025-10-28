/**
 * Emergency script to repair racing session telemetry ordering.
 *
 * Existing recordings may contain telemetry entries stored out of order across
 * GridFS chunks. The racing-session service expects the data to be sorted by
 * `Physics_packed_id`, so this script rebuilds the chunks in the correct order.
 *
 * Usage examples:
 *   npx ts-node src/scripts/fix-session-order.ts --session <sessionId>
 *   npx ts-node src/scripts/fix-session-order.ts --dry-run
 *
 * Required environment variables (identical to the Nest application):
 *   MONGO_CLIENTNAME, MONGO_CLIENTPASSWORD, MONGO_URL
 *   Alternatively supply --mongo-uri to override the connection string.
 */

import mongoose, { HydratedDocument, Model, Types } from 'mongoose';
import { GridFSBucket, ObjectId as MongoObjectId } from 'mongodb';
import { RacingSession, RacingSessionSchema } from '../schemas/racing-session.schema';

type RacingSessionDocument = HydratedDocument<RacingSession>;

interface ScriptOptions {
    sessionId?: string;
    dryRun: boolean;
    limit?: number;
    mongoUri?: string;
}

interface SortResult {
    sortedRecords: any[];
    hasOrderChanges: boolean;
    stats: {
        totalRecords: number;
        missingPackedId: number;
    };
}

const RACING_SESSIONS_BUCKET = 'racing_sessions';

function parseArgs(): ScriptOptions {
    const args = process.argv.slice(2);
    const options: ScriptOptions = {
        dryRun: false
    };

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        switch (arg) {
            case '--session':
            case '--sessionId':
                options.sessionId = args[++i];
                break;
            case '--limit':
                options.limit = Number.parseInt(args[++i], 10);
                break;
            case '--dry-run':
            case '--dryRun':
                options.dryRun = true;
                break;
            case '--mongo-uri':
            case '--mongoUri':
                options.mongoUri = args[++i];
                break;
            case '--help':
            case '-h':
                printHelp();
                process.exit(0);
                break;
            default:
                console.warn(`Unknown argument '${arg}' ignored.`);
        }
    }

    return options;
}

function printHelp(): void {
    console.log(`\nUsage: ts-node src/scripts/fix-session-order.ts [options]\n\n` +
        `Options:\n` +
        `  --session <id>     Repair a single racing session by Mongo ObjectId\n` +
        `  --limit <count>    Stop after processing <count> sessions\n` +
        `  --dry-run          Analyse sessions without writing changes\n` +
        `  --mongo-uri <uri>  Override the MongoDB connection string\n` +
        `  --help             Show this message\n`);
}

function resolveMongoUri(override?: string): string {
    if (override) {
        return override;
    }

    if (process.env.MONGO_URI) {
        return process.env.MONGO_URI;
    }

    const username = process.env.MONGO_CLIENTNAME;
    const password = process.env.MONGO_CLIENTPASSWORD;
    const host = process.env.MONGO_URL;

    if (username && password && host) {
        const encodedUser = encodeURIComponent(username);
        const encodedPassword = encodeURIComponent(password);
        return `mongodb://${encodedUser}:${encodedPassword}@${host}:27017/ACLA`;
    }

    throw new Error('Unable to resolve MongoDB connection string. Provide --mongo-uri or set MONGO_URI / MONGO_CLIENTNAME / MONGO_CLIENTPASSWORD / MONGO_URL.');
}

async function readJsonChunk(bucket: GridFSBucket, fileId: MongoObjectId): Promise<any[]> {
    return new Promise<any[]>((resolve, reject) => {
        const downloadStream = bucket.openDownloadStream(fileId);
        const chunks: Buffer[] = [];

        downloadStream.on('data', (chunk) => {
            chunks.push(chunk);
        });

        downloadStream.on('error', (error) => {
            reject(new Error(`Failed to download chunk ${fileId.toHexString()}: ${error.message}`));
        });

        downloadStream.on('end', () => {
            try {
                const jsonString = Buffer.concat(chunks).toString('utf8');
                const parsed = JSON.parse(jsonString);
                if (Array.isArray(parsed)) {
                    resolve(parsed);
                } else {
                    reject(new Error(`Chunk ${fileId.toHexString()} is not a JSON array.`));
                }
            } catch (error) {
                reject(new Error(`Failed to parse chunk ${fileId.toHexString()}: ${(error as Error).message}`));
            }
        });
    });
}

function toMongoObjectId(value: Types.ObjectId | MongoObjectId | string): MongoObjectId {
    if (value instanceof MongoObjectId) {
        return value;
    }

    if (value && typeof (value as any).toHexString === 'function') {
        return new MongoObjectId((value as any).toHexString());
    }

    if (typeof value === 'string') {
        if (!Types.ObjectId.isValid(value)) {
            throw new Error(`Invalid ObjectId value: ${value}`);
        }
        return new MongoObjectId(value);
    }

    throw new Error('Unsupported ObjectId value provided.');
}

function decorateRecords(records: any[]) {
    return records.map((record, originalIndex) => {
        const rawValue = record?.Physics_packed_id;
        const numericValue = typeof rawValue === 'number' ? rawValue : Number(rawValue);
        const hasNumeric = Number.isFinite(numericValue);

        return {
            row: record,
            sortValue: hasNumeric ? numericValue : Number.POSITIVE_INFINITY,
            hasNumeric,
            originalIndex
        };
    });
}

function sortRecords(records: any[]): SortResult {
    const decorated = decorateRecords(records);

    const sortedDecorated = [...decorated].sort((a, b) => {
        if (a.sortValue === b.sortValue) {
            if (a.hasNumeric !== b.hasNumeric) {
                return a.hasNumeric ? -1 : 1;
            }
            return a.originalIndex - b.originalIndex;
        }
        return a.sortValue - b.sortValue;
    });

    const sortedRecords = sortedDecorated.map((item) => item.row);
    const hasOrderChanges = sortedRecords.some((item, index) => item !== records[index]);

    return {
        sortedRecords,
        hasOrderChanges,
        stats: {
            totalRecords: records.length,
            missingPackedId: decorated.filter((item) => !item.hasNumeric).length
        }
    };
}

function chunkArray<T>(items: T[], chunkSize: number): T[][] {
    if (chunkSize <= 0) {
        throw new Error(`Chunk size must be positive. Received: ${chunkSize}`);
    }

    const chunks: T[][] = [];

    for (let index = 0; index < items.length; index += chunkSize) {
        chunks.push(items.slice(index, index + chunkSize));
    }

    return chunks;
}

async function uploadChunk(bucket: GridFSBucket, data: any[], filename: string, metadata: Record<string, unknown>): Promise<MongoObjectId> {
    return new Promise<MongoObjectId>((resolve, reject) => {
        const uploadStream = bucket.openUploadStream(filename, { metadata });

        uploadStream.on('error', (error) => {
            reject(new Error(`Failed to upload chunk '${filename}': ${error.message}`));
        });

        uploadStream.on('finish', () => {
            resolve(uploadStream.id as MongoObjectId);
        });

        uploadStream.end(Buffer.from(JSON.stringify(data)));
    });
}

async function deleteChunk(bucket: GridFSBucket, fileId: MongoObjectId): Promise<void> {
    try {
        await bucket.delete(fileId);
    } catch (error) {
        console.warn(`Failed to delete old chunk ${fileId.toHexString()}: ${(error as Error).message}`);
    }
}

async function processSession(
    session: RacingSessionDocument,
    bucket: GridFSBucket,
    options: ScriptOptions
): Promise<{ updated: boolean; stats?: SortResult['stats']; }> {
    const rawChunkIds = session.dataChunkFileIds ? Array.from(session.dataChunkFileIds) : [];
    const chunkIds = rawChunkIds.map((id) => toMongoObjectId(id as unknown as Types.ObjectId));

    if (chunkIds.length === 0) {
        console.log(`Session ${session._id.toHexString()} has no chunk files; skipping.`);
        return { updated: false };
    }

    const combinedData: any[] = [];

    for (const fileId of chunkIds) {
        const chunk = await readJsonChunk(bucket, fileId);
        combinedData.push(...chunk);
    }

    if (combinedData.length === 0) {
        console.log(`Session ${session._id.toHexString()} contains no telemetry entries; skipping.`);
        return { updated: false };
    }

    const sortResult = sortRecords(combinedData);

    if (!sortResult.hasOrderChanges) {
        console.log(`Session ${session._id.toHexString()} is already ordered; no changes needed.`);
        return { updated: false, stats: sortResult.stats };
    }

    console.log(`Session ${session._id.toHexString()} requires reordering (${sortResult.stats.totalRecords} records, ${sortResult.stats.missingPackedId} missing Physics_packed_id).`);

    if (options.dryRun) {
        console.log('Dry-run mode active; skipping write operations.');
        return { updated: false, stats: sortResult.stats };
    }

    const chunkSize = session.chunkSize && session.chunkSize > 0 ? session.chunkSize : 1000;
    const sortedChunks = chunkArray(sortResult.sortedRecords, chunkSize);
    const newChunkIds: MongoObjectId[] = [];

    for (let index = 0; index < sortedChunks.length; index++) {
        const chunk = sortedChunks[index];
        if (chunk.length === 0) {
            continue;
        }

        const filename = `session_${session.session_name}_${session.map}_${session.car_name}_chunk_${index}_${Date.now()}.json`;
        const metadata = {
            session_name: session.session_name,
            map: session.map,
            car_name: session.car_name,
            userId: session.user_id,
            chunkIndex: index,
            totalChunks: sortedChunks.length,
            chunkSize,
            repairedAt: new Date(),
            repairedBy: 'fix-session-order.ts'
        };

        const fileId = await uploadChunk(bucket, chunk, filename, metadata);
        newChunkIds.push(fileId);
    }

    if (newChunkIds.length === 0) {
        throw new Error(`No replacement chunks were created for session ${session._id.toHexString()}.`);
    }

    session.set({
        dataChunkFileIds: newChunkIds.map((id) => new Types.ObjectId(id.toHexString())),
        chunkSize,
        totalChunks: newChunkIds.length,
        totalDataPoints: sortResult.sortedRecords.length
    });

    await session.save();

    for (const oldId of chunkIds) {
        await deleteChunk(bucket, oldId);
    }

    console.log(`Session ${session._id.toHexString()} reordered successfully (chunks: ${newChunkIds.length}).`);

    return { updated: true, stats: sortResult.stats };
}

async function main(): Promise<void> {
    const options = parseArgs();
    const mongoUri = resolveMongoUri(options.mongoUri);

    await mongoose.connect(mongoUri, {
        serverSelectionTimeoutMS: 30000,
        appName: 'fix-session-order'
    });

    const connection = mongoose.connection;
    if (!connection.db) {
        throw new Error('MongoDB connection established without database handle.');
    }

    const bucket = new GridFSBucket(connection.db, { bucketName: RACING_SESSIONS_BUCKET });

    const RacingSessionModel: Model<RacingSession> = connection.model<RacingSession>(
        'RacingSession',
        RacingSessionSchema,
        'racingsessions'
    );

    const filter: Record<string, unknown> = {};
    if (options.sessionId) {
        if (!Types.ObjectId.isValid(options.sessionId)) {
            throw new Error(`Invalid session id provided: ${options.sessionId}`);
        }
        filter._id = new Types.ObjectId(options.sessionId);
    }

    const cursor = RacingSessionModel.find(filter).cursor();

    let processed = 0;
    let updated = 0;
    let skipped = 0;

    for (let session = await cursor.next(); session !== null; session = await cursor.next()) {
        processed += 1;

        if (options.limit && processed > options.limit) {
            console.log('Processing limit reached; stopping early.');
            break;
        }

        console.log(`\nProcessing session ${session._id.toHexString()} (${session.session_name} / ${session.map} / ${session.car_name})`);
        try {
            const result = await processSession(session, bucket, options);
            if (result.updated) {
                updated += 1;
            } else {
                skipped += 1;
            }
        } catch (error) {
            console.error(`Failed to repair session ${session._id.toHexString()}: ${(error as Error).message}`);
        }
    }

    await cursor.close();

    console.log(`\nCompleted. Sessions processed: ${processed}, reordered: ${updated}, skipped: ${skipped}.`);

    await mongoose.disconnect();
}

main().catch((error) => {
    console.error('Fatal error:', error instanceof Error ? error.message : error);
    mongoose.disconnect().finally(() => process.exit(1));
});
