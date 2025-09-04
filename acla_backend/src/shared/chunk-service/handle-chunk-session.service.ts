import { Injectable, Logger } from '@nestjs/common';
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

    constructor() {
        // Clean up expired sessions every 10 minutes
        setInterval(() => {
            this.cleanupExpiredSessions();
        }, 10 * 60 * 1000);
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

            // Store chunk
            session.chunks.set(chunkIndex, data);
            session.receivedChunks = session.chunks.size;
            session.lastUpdated = new Date();

            this.logger.debug(
                `Received chunk ${chunkIndex + 1}/${totalChunks} for session ${sessionId} (${session.receivedChunks}/${totalChunks} total)`
            );

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
            this.logger.error(`Error processing chunk for session ${sessionId}:`, error.message);

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

        // Assemble data in correct order
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
        const deleted = this.sessions.delete(sessionId);
        if (deleted) {
            this.logger.debug(`Cleaned up session ${sessionId}`);
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
        return {
            sessionId,
            totalChunks,
            receivedChunks: 0,
            chunks: new Map(),
            createdAt: new Date(),
            lastUpdated: new Date(),
            status: 'pending',
        };
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
            this.sessions.delete(sessionId);
            this.logger.warn(`Cleaned up expired session: ${sessionId}`);
        });

        if (expiredSessions.length > 0) {
            this.logger.log(`Cleaned up ${expiredSessions.length} expired sessions`);
        }
    }
}
