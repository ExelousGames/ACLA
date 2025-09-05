import { Injectable, Logger, HttpStatus } from '@nestjs/common';
import { HttpException } from '@nestjs/common';
import {
    ChunkData,
    ChunkPrepareOptions,
    ChunkSessionStatus,
} from './interfaces/chunk.interface';
import { ChunkHandlerService } from './chunk-handler.service';
import { HandleChunkSessionService } from './handle-chunk-session.service';

/**
 * Main service for handling chunked JSON requests
 * This service provides a high-level interface for other controllers
 * to easily implement chunked data transfer functionality
 */
@Injectable()
export class ChunkClientService {
    private readonly logger = new Logger(ChunkClientService.name);

    constructor(
        private readonly chunkHandler: ChunkHandlerService,
        private readonly chunkRequestHandler: HandleChunkSessionService
    ) { }

    /**
     * Controller helper: Process incoming chunked data
     * Use this in your controller to handle chunked POST requests
     */
    async handleIncomingChunk<T>(
        chunkData: ChunkData,
        processCallback: (data: any) => Promise<T>
    ) {
        try {
            this.logger.debug(
                `Handling incoming chunk ${chunkData.chunkIndex + 1}/${chunkData.totalChunks} ` +
                `for session ${chunkData.sessionId}`
            );

            const result = await this.chunkHandler.processChunkedData(chunkData, processCallback);

            return {
                success: true,
                sessionId: chunkData.sessionId,
                chunkIndex: chunkData.chunkIndex,
                totalChunks: chunkData.totalChunks,
                isComplete: result.response.isComplete,
                receivedChunks: result.response.receivedChunks,
                message: result.response.message,
                data: result.processedResult || null,
            };
        } catch (error) {
            this.logger.error(`Error handling incoming chunk:`, error.message);
            throw new HttpException(
                {
                    success: false,
                    message: `Failed to process chunk: ${error.message}`,
                    sessionId: chunkData.sessionId,
                },
                HttpStatus.BAD_REQUEST
            );
        }
    }

    /**
     * Controller helper: Prepare data for chunked sending
     * Use this in your controller to prepare large responses for chunked transfer
     */
    async prepareDataForChunkedSending(data: any, chunkSize?: number) {
        try {
            const options: ChunkPrepareOptions = {
                data: data,
                chunkSize,
            };

            // Prepare the chunks
            const result = await this.chunkHandler.prepareChunks(options);

            this.logger.log(
                `Prepared data for chunked sending: ${result.totalChunks} chunks, session ${result.sessionId}`
            );

            return {
                success: true,
                sessionId: result.sessionId,
                totalChunks: result.totalChunks,
                chunks: result.chunks,
                message: `Data prepared for chunked transfer (${result.totalChunks} chunks)`,
            };
        } catch (error) {
            this.logger.error('Error preparing data for chunked sending:', error.message);
            throw new HttpException(
                {
                    success: false,
                    message: `Failed to prepare chunked data: ${error.message}`,
                },
                HttpStatus.INTERNAL_SERVER_ERROR
            );
        }
    }

    /**
     * Controller helper: Get upload/download status
     * Use this in your controller to provide status endpoints
     */
    async getSessionStatus(sessionId: string) {
        try {
            const status = await this.chunkHandler.getSessionStatus(sessionId);
            return {
                success: true,
                ...status,
            };
        } catch (error) {
            this.logger.error(`Error getting session status for ${sessionId}:`, error.message);
            throw new HttpException(
                {
                    success: false,
                    message: error.message || 'Session not found',
                    sessionId,
                },
                HttpStatus.NOT_FOUND
            );
        }
    }

    /**
     * Controller helper: Get all active sessions
     * Use this for monitoring/admin endpoints
     */
    async getAllActiveSessions() {
        try {
            const sessions = await this.chunkHandler.getActiveSessions();
            return {
                success: true,
                sessions,
                count: sessions.length,
            };
        } catch (error) {
            this.logger.error('Error getting active sessions:', error.message);
            throw new HttpException(
                {
                    success: false,
                    message: 'Failed to retrieve active sessions',
                },
                HttpStatus.INTERNAL_SERVER_ERROR
            );
        }
    }

    /**
     * Controller helper: Clean up a session
     * Use this to manually clean up completed or failed sessions
     */
    async cleanupSession(sessionId: string) {
        try {
            const deleted = await this.chunkHandler.cleanupSession(sessionId);
            return {
                success: true,
                message: deleted ? 'Session cleaned up successfully' : 'Session not found',
                sessionId,
                deleted,
            };
        } catch (error) {
            this.logger.error(`Error cleaning up session ${sessionId}:`, error.message);
            throw new HttpException(
                {
                    success: false,
                    message: `Failed to cleanup session: ${error.message}`,
                    sessionId,
                },
                HttpStatus.INTERNAL_SERVER_ERROR
            );
        }
    }

    /**
     * Utility: Estimate chunk requirements for data
     * Use this to help clients determine optimal chunk sizes
     */
    estimateChunkRequirements(data: any, chunkSize?: number) {
        try {
            const estimates = this.chunkHandler.estimateChunkRequirements(data, chunkSize);
            return {
                success: true,
                ...estimates,
                humanReadableSize: this.formatBytes(estimates.estimatedSize),
            };
        } catch (error) {
            this.logger.error('Error estimating chunk requirements:', error.message);
            return {
                success: false,
                message: `Failed to estimate requirements: ${error.message}`,
            };
        }
    }

    /**
     * Client helper: Send data in chunks to an endpoint
     * Use this method when you need to send large data from one service to another
     */
    async sendDataInChunks(
        data: any,
        sendFunction: (chunk: ChunkData) => Promise<any>,
        options?: {
            chunkSize?: number;
            onProgress?: (progress: { current: number; total: number; percentage: number }) => void;
        }
    ) {
        try {
            // First prepare the chunks
            const prepareResult = await this.chunkHandler.prepareChunks({
                data: data,
                chunkSize: options?.chunkSize,
            });

            this.logger.log(`Sending ${prepareResult.totalChunks} chunks for session ${prepareResult.sessionId}`);

            // Then send them
            const results = await this.chunkHandler.sendChunkedData(
                prepareResult.chunks,
                sendFunction,
                options?.onProgress
            );

            return {
                success: true,
                sessionId: prepareResult.sessionId,
                totalChunks: prepareResult.totalChunks,
                results,
                message: 'All chunks sent successfully',
            };
        } catch (error) {
            this.logger.error('Error sending data in chunks:', error.message);
            throw new HttpException(
                {
                    success: false,
                    message: `Failed to send chunked data: ${error.message}`,
                },
                HttpStatus.INTERNAL_SERVER_ERROR
            );
        }
    }

    /**
     * Get a specific prepared chunk from a session
     * Use this to retrieve individual chunks from a prepared session
     */
    async getPreparedChunk(sessionId: string, chunkIndex: number) {
        try {
            const chunk = this.chunkHandler.getPreparedChunk(sessionId, chunkIndex);
            const sessionStatus = await this.getSessionStatus(sessionId);

            return {
                success: true,
                sessionId,
                chunkIndex,
                totalChunks: sessionStatus.totalChunks,
                data: chunk,
                isLastChunk: chunkIndex === sessionStatus.totalChunks - 1,
            };
        } catch (error) {
            this.logger.error(`Error getting prepared chunk ${chunkIndex} from session ${sessionId}:`, error.message);
            throw new HttpException(
                {
                    success: false,
                    message: error.message || 'Chunk not found',
                    sessionId,
                    chunkIndex,
                },
                HttpStatus.NOT_FOUND
            );
        }
    }

    /**
     * Format bytes to human readable format
     */
    private formatBytes(bytes: number, decimals = 2): string {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
}