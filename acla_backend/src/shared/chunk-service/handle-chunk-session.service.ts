import { Injectable, Logger } from '@nestjs/common';
import { promises as fsPromises, createReadStream, createWriteStream } from 'fs';
import * as path from 'path';
import { Readable } from 'stream';
import { pipeline as pipelinePromise } from 'stream/promises';
import {
    ChunkData,
    ChunkSession,
    ChunkSessionStatus,
    ChunkProcessResult,
} from './interfaces/chunk.interface';

/**
 * manage chunks together
 * each chunk is processed and tracked within a session
 */
@Injectable()
export class HandleChunkSessionService {
    private readonly logger = new Logger(HandleChunkSessionService.name);
    private readonly sessions = new Map<string, ChunkSession>();
    private readonly SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes
    // store large chunks on disk to avoid excessive memory usage
    private readonly LARGE_CHUNK_THRESHOLD_BYTES = 1 * 1024 * 1024; // 1 MiB (tune as needed)
    private readonly BASE_TMP_DIR = path.resolve(process.cwd(), 'session_recording', 'temp', 'chunk-sessions');

    constructor() {
        // Clean up expired sessions every 10 minutes
        setInterval(() => {
            this.cleanupExpiredSessions();
        }, 10 * 60 * 1000);

        // Ensure base temp dir exists
        this.ensureDir(this.BASE_TMP_DIR).catch((err) => {
            this.logger.error(`Failed to create base temp dir ${this.BASE_TMP_DIR}: ${err?.message || err}`);
        });
    }

    /**
     * Process received chunk data
     */
    async processChunk(chunkData: ChunkData): Promise<ChunkProcessResult> {
        const { sessionId, chunkIndex, totalChunks, data } = chunkData;

        try {
            // Get or create session
            let session = this.sessions.get(sessionId);
            if (!session) {
                session = this.createSession(sessionId, totalChunks);
                this.sessions.set(sessionId, session);
            }

            // Validate chunk
            if (chunkIndex >= totalChunks || chunkIndex < 0) {
                throw new Error(`Invalid chunk index ${chunkIndex} for total chunks ${totalChunks}`);
            }

            if (session.totalChunks !== totalChunks) {
                throw new Error(`Chunk total mismatch: expected ${session.totalChunks}, got ${totalChunks}`);
            }

            // Store chunk (disk-backed if large)
            const storedRef = await this.storeChunk(session, chunkIndex, data);
            session.chunks.set(chunkIndex, storedRef);
            session.receivedChunks = session.chunks.size;
            session.lastUpdated = new Date();

            const isComplete = session.receivedChunks === totalChunks;

            if (isComplete) {
                session.status = 'complete';
                this.logger.log(`Session ${sessionId} completed with all ${totalChunks} chunks received`);
            }

            return {
                response: {
                    success: true,
                    message: isComplete
                        ? 'All chunks received and processed successfully'
                        : `Chunk ${chunkIndex + 1}/${totalChunks} received successfully`,
                    sessionId,
                    isComplete,
                    receivedChunks: session.receivedChunks,
                    totalChunks: session.totalChunks,
                },
            };
        } catch (error) {
            this.logger.error(`Error processing chunk for session ${sessionId}:`, (error as any)?.message || error);

            const session = this.sessions.get(sessionId);
            if (session) {
                session.status = 'failed';
            }

            return {
                response: {
                    success: false,
                    message: error.message || 'Failed to process chunk',
                    sessionId,
                    isComplete: false,
                    receivedChunks: session?.receivedChunks || 0,
                    totalChunks: session?.totalChunks || totalChunks,
                },
            };
        }
    }

    /**
     * Get complete assembled data from all chunks
     */
    getAssembledData(sessionId: string): any {
        const session = this.sessions.get(sessionId);

        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        if (session.status !== 'complete') {
            throw new Error(`Session ${sessionId} is not complete (${session.receivedChunks}/${session.totalChunks} chunks received)`);
        }

        // If any chunk is disk-backed, advise using streaming APIs
        const hasDiskBacked = Array.from(session.chunks.values()).some((v: any) => v && typeof v === 'object' && v.filePath);
        if (hasDiskBacked) {
            throw new Error(
                `Session ${sessionId} uses disk-backed storage; use getAssembledStream() or assembleToFile() for memory-safe assembly.`
            );
        }

        // Assemble data in correct order (memory-only sessions). Note: For very large payloads prefer stream APIs.
        const assembledData: any[] = [];
        for (let i = 0; i < session.totalChunks; i++) {
            const chunk = session.chunks.get(i);
            if (!chunk) {
                throw new Error(`Missing chunk ${i} for session ${sessionId}`);
            }
            assembledData.push(chunk);
        }

        this.logger.debug(`Assembled data for session ${sessionId} with ${assembledData.length} chunks`);
        return assembledData;
    }

    /**
     * Get session status
     */
    getSessionStatus(sessionId: string): ChunkSessionStatus {
        const session = this.sessions.get(sessionId);

        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        const completionPercentage = (session.receivedChunks / session.totalChunks) * 100;

        // Estimate remaining time based on received chunks rate
        let estimatedTimeRemaining: number | undefined;
        if (session.receivedChunks > 0 && session.status === 'pending') {
            const elapsedTime = Date.now() - session.createdAt.getTime();
            const avgTimePerChunk = elapsedTime / session.receivedChunks;
            const remainingChunks = session.totalChunks - session.receivedChunks;
            estimatedTimeRemaining = Math.round(avgTimePerChunk * remainingChunks);
        }

        return {
            sessionId: session.sessionId,
            status: session.status,
            totalChunks: session.totalChunks,
            receivedChunks: session.receivedChunks,
            completionPercentage: Math.round(completionPercentage * 100) / 100,
            createdAt: session.createdAt,
            lastUpdated: session.lastUpdated,
            estimatedTimeRemaining,
        };
    }

    /**
     * Clean up session after processing
     */
    cleanupSession(sessionId: string): boolean {
        const session = this.sessions.get(sessionId);
        const deleted = this.sessions.delete(sessionId);
        if (deleted) {
            this.logger.debug(`Cleaned up session ${sessionId}`);
            // remove any temp files associated with this session
            if (session?.metadata?.sessionDir) {
                this.safeRemoveDir(session.metadata.sessionDir).catch((err) =>
                    this.logger.warn(`Failed to remove temp dir for ${sessionId}: ${err?.message || err}`)
                );
            }
        }
        return deleted;
    }

    /**
     * Get all active sessions
     */
    getActiveSessions(): ChunkSessionStatus[] {
        return Array.from(this.sessions.values()).map(session => ({
            sessionId: session.sessionId,
            status: session.status,
            totalChunks: session.totalChunks,
            receivedChunks: session.receivedChunks,
            completionPercentage: Math.round((session.receivedChunks / session.totalChunks) * 10000) / 100,
            createdAt: session.createdAt,
            lastUpdated: session.lastUpdated,
        }));
    }

    /**
     * Create a new session
     */
    private createSession(sessionId: string, totalChunks: number): ChunkSession {
        const sessionDir = path.join(this.BASE_TMP_DIR, sessionId);
        // create a per-session dir lazily (on first large chunk) but we can create upfront for simplicity
        this.ensureDir(sessionDir).catch((err) =>
            this.logger.warn(`Could not pre-create session dir ${sessionDir}: ${err?.message || err}`)
        );

        return {
            sessionId,
            totalChunks,
            receivedChunks: 0,
            chunks: new Map(),
            createdAt: new Date(),
            lastUpdated: new Date(),
            status: 'pending',
            metadata: {
                sessionDir,
            },
        };
    }

    /**
     * Store prepared chunks for a session (for chunked sending)
     */
    storePreparedChunks(sessionId: string, chunks: ChunkData[]): void {
        const session: ChunkSession = {
            sessionId,
            totalChunks: chunks.length,
            receivedChunks: chunks.length, // All chunks are already prepared
            chunks: new Map<number, any>(),
            createdAt: new Date(),
            lastUpdated: new Date(),
            status: 'complete',
            metadata: { isPreparedForSending: true }
        };

        // Store all chunks
        chunks.forEach(chunk => {
            session.chunks.set(chunk.chunkIndex, chunk.data);
        });

        this.sessions.set(sessionId, session);
        this.logger.debug(`Stored ${chunks.length} prepared chunks for session ${sessionId}`);
    }

    /**
     * Get a specific chunk from a prepared session
     */
    getPreparedChunk(sessionId: string, chunkIndex: number): any {
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        if (!session.metadata?.isPreparedForSending) {
            throw new Error(`Session ${sessionId} is not prepared for sending`);
        }

        if (chunkIndex < 0 || chunkIndex >= session.totalChunks) {
            throw new Error(`Invalid chunk index ${chunkIndex} for session ${sessionId}`);
        }

        const chunk = session.chunks.get(chunkIndex);
        if (chunk === undefined) {
            throw new Error(`Chunk ${chunkIndex} not found in session ${sessionId}`);
        }

        return chunk;
    }

    /**
     * Clean up expired sessions
     */
    private cleanupExpiredSessions(): void {
        const now = Date.now();
        const expiredSessions: string[] = [];

        this.sessions.forEach((session, sessionId) => {
            const isExpired = now - session.lastUpdated.getTime() > this.SESSION_TIMEOUT;
            if (isExpired) {
                expiredSessions.push(sessionId);
            }
        });

        expiredSessions.forEach(sessionId => {
            const s = this.sessions.get(sessionId);
            this.sessions.delete(sessionId);
            // best-effort cleanup of temp files
            if (s?.metadata?.sessionDir) {
                this.safeRemoveDir(s.metadata.sessionDir).catch((err) =>
                    this.logger.warn(`Failed to remove temp dir for expired session ${sessionId}: ${err?.message || err}`)
                );
            }
            this.logger.warn(`Cleaned up expired session: ${sessionId}`);
        });

        if (expiredSessions.length > 0) {
            this.logger.log(`Cleaned up ${expiredSessions.length} expired sessions`);
        }
    }

    /**
     * Create a readable stream that emits the assembled content in order without buffering it all in memory.
     * Works for both memory-only and disk-backed chunks.
     */
    getAssembledStream(sessionId: string): Readable {
        const session = this.sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }
        if (session.status !== 'complete') {
            throw new Error(
                `Session ${sessionId} is not complete (${session.receivedChunks}/${session.totalChunks} chunks received)`
            );
        }

        // Build a composite stream from all chunks in order
        const s = session as ChunkSession; // capture validated session for type narrowing in generator
        async function* generator() {
            for (let i = 0; i < s.totalChunks; i++) {
                const ref = s.chunks.get(i);
                if (!ref) {
                    throw new Error(`Missing chunk ${i} for session ${sessionId}`);
                }
                // disk-backed
                if (ref && typeof ref === 'object' && ref.filePath) {
                    const stream = createReadStream(ref.filePath);
                    for await (const chunk of stream) {
                        yield chunk as Buffer;
                    }
                } else {
                    // in-memory buffer/string
                    yield ref;
                }
            }
        }

        return Readable.from(generator());
    }

    /**
     * Assemble into a single file at outputPath using streaming, returning the final file path.
     */
    async assembleToFile(sessionId: string, outputPath: string): Promise<string> {
        const dir = path.dirname(outputPath);
        await this.ensureDir(dir);
        const readable = this.getAssembledStream(sessionId);
        const writable = createWriteStream(outputPath);
        await pipelinePromise(readable, writable);
        this.logger.debug(`Assembled session ${sessionId} into file ${outputPath}`);
        return outputPath;
    }

    /**
     * Helpers
     */
    private async ensureDir(dirPath: string): Promise<void> {
        await fsPromises.mkdir(dirPath, { recursive: true });
    }

    private async writeChunkToFile(filePath: string, data: any): Promise<number> {
        // Normalize data to Buffer or string accepted by write stream
        return new Promise<number>((resolve, reject) => {
            const ws = createWriteStream(filePath, { flags: 'w' });
            let total = 0;
            ws.on('error', reject);
            ws.on('finish', () => resolve(total));
            if (data instanceof Readable) {
                data.on('data', (d: Buffer) => (total += d.length));
                data.pipe(ws);
            } else {
                const buf = Buffer.isBuffer(data) ? data : Buffer.from(String(data));
                total = buf.length;
                ws.end(buf);
            }
        });
    }

    private estimateByteLength(data: any): number {
        if (Buffer.isBuffer(data)) return data.length;
        if (typeof data === 'string') return Buffer.byteLength(data);
        // try JSON stringify as last resort
        try {
            return Buffer.byteLength(JSON.stringify(data));
        } catch {
            return 0;
        }
    }

    private async storeChunk(
        session: ChunkSession,
        chunkIndex: number,
        data: any
    ): Promise<any /* Buffer|string|{filePath:string,size:number} */> {
        const size = this.estimateByteLength(data);
        // if above threshold, write to disk
        if (size >= this.LARGE_CHUNK_THRESHOLD_BYTES) {
            const sessionDir: string = session.metadata?.sessionDir || path.join(this.BASE_TMP_DIR, session.sessionId);
            await this.ensureDir(sessionDir);
            if (!session.metadata) session.metadata = {} as any;
            session.metadata.sessionDir = sessionDir;
            const filePath = path.join(sessionDir, `${chunkIndex}.chunk`);
            await this.writeChunkToFile(filePath, data);
            return { filePath, size };
        }
        // small chunk: keep in memory
        return data;
    }

    private async safeRemoveDir(dirPath: string): Promise<void> {
        try {
            await fsPromises.rm(dirPath, { recursive: true, force: true });
        } catch (e) {
            // ignore
        }
    }
}
