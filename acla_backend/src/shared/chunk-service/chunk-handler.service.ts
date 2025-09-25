import { Injectable, Logger } from '@nestjs/common';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import {
    ChunkData,
    ChunkPrepareOptions,
    ChunkPrepareResult,
    ChunkProcessResult,
    ChunkSessionStatus,
} from './interfaces/chunk.interface';
import { HandleChunkSessionService } from './handle-chunk-session.service';

/**
 * Service for handling each chunked data preparation and processing
 */
@Injectable()
export class ChunkHandlerService {
    private readonly logger = new Logger(ChunkHandlerService.name);
    private readonly DEFAULT_CHUNK_SIZE = 1024 * 1024; // 1MB default chunk size

    constructor(
        private readonly handleChunkSessionService: HandleChunkSessionService
    ) { }

    /**
     * Prepare large data for chunked sending. serialize and split into chunks
     * @param options 
     * @returns
     */
    async prepareChunks(options: ChunkPrepareOptions): Promise<ChunkPrepareResult> {
        const { data: content, chunkSize = this.DEFAULT_CHUNK_SIZE, metadata } = options;

        try {
            // Serialize content to JSON string
            const serializedContent = JSON.stringify(content);
            const contentSize = Buffer.byteLength(serializedContent, 'utf8');

            this.logger.debug(`Preparing chunks for content of size: ${contentSize} bytes`);

            // Calculate number of chunks needed
            const totalChunks = Math.ceil(contentSize / chunkSize);
            const sessionId = uuidv4();

            const chunks: ChunkData[] = [];

            // Split content into chunks
            for (let i = 0; i < totalChunks; i++) {
                const start = i * chunkSize;
                const end = Math.min(start + chunkSize, contentSize);
                const chunkContent = serializedContent.slice(start, end);

                //data is split into chunks and prepared for processing
                const chunkData: ChunkData = {
                    sessionId,
                    chunkIndex: i,
                    totalChunks,
                    data: chunkContent,
                    metadata: {
                        timestamp: new Date(),
                        size: Buffer.byteLength(chunkContent, 'utf8'),
                        ...metadata,
                    },
                };

                chunks.push(chunkData);
            }

            this.logger.log(`Prepared ${totalChunks} chunks for session ${sessionId}`);

            // Store the prepared chunks for later retrieval
            this.handleChunkSessionService.storePreparedChunks(sessionId, chunks);

            return {
                sessionId,
                totalChunks,
                chunks,
            };
        } catch (error) {
            this.logger.error('Error preparing chunks:', error.message);
            throw new Error(`Failed to prepare chunks: ${error.message}`);
        }
    }

    /**
     * Process received chunked data and execute callback when complete
     */
    async processChunkedData<T>(
        chunkData: ChunkData,
        processCallback: (assembledData: any) => Promise<T>,
        options?: { assemblyMode?: 'json' | 'buffer' | 'stream' | 'file'; outputPath?: string }
    ): Promise<ChunkProcessResult> {
        try {

            // Process the chunk
            const result = await this.handleChunkSessionService.processChunk(chunkData);

            // If all chunks are received, assemble and process the data
            if (result.response.isComplete) {
                try {
                    const mode = options?.assemblyMode || 'json';
                    let processedResult: T;

                    if (mode === 'stream') {
                        const stream = this.handleChunkSessionService.getAssembledStream(chunkData.sessionId);
                        this.logger.log(`Assembling session ${chunkData.sessionId} as stream`);
                        processedResult = await processCallback(stream);
                    } else if (mode === 'file') {
                        const outputPath = options?.outputPath || path.resolve(process.cwd(), 'session_recording', 'temp', 'assembled', `${chunkData.sessionId}.bin`);
                        const finalPath = await this.handleChunkSessionService.assembleToFile(chunkData.sessionId, outputPath);
                        this.logger.log(`Assembled session ${chunkData.sessionId} to file ${finalPath}`);
                        processedResult = await processCallback(finalPath);
                    } else if (mode === 'buffer') {
                        // Caution: may consume significant memory for very large payloads
                        const readStream = this.handleChunkSessionService.getAssembledStream(chunkData.sessionId);
                        const chunks: Buffer[] = [];
                        for await (const c of readStream) {
                            chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(String(c)));
                        }
                        const buf = Buffer.concat(chunks);
                        processedResult = await processCallback(buf);
                    } else {
                        // Default JSON path (legacy behavior). May fail for extremely large strings.
                        const assembledChunks = this.handleChunkSessionService.getAssembledData(chunkData.sessionId);
                        const reconstructedContent = assembledChunks.join('');
                        const originalData = JSON.parse(reconstructedContent);
                        this.logger.log(`Successfully assembled JSON data for session ${chunkData.sessionId}`);
                        processedResult = await processCallback(originalData);
                    }

                    // Clean up the session after successful processing
                    this.handleChunkSessionService.cleanupSession(chunkData.sessionId);

                    return {
                        ...result,
                        processedResult,
                    };
                } catch (error) {
                    this.logger.error(`Error processing assembled data for session ${chunkData.sessionId}:`, error.message);
                    // Provide guidance if default JSON path failed due to size
                    if (String(error?.message || '').includes('disk-backed') || String(error?.message || '').includes('Invalid string length')) {
                        throw new Error(
                            `Failed to process assembled data: ${error.message}. ` +
                            `For very large payloads, call processChunkedData with options { assemblyMode: 'stream' } or { assemblyMode: 'file' } to avoid loading content into memory as a single string.`
                        );
                    }
                    throw new Error(`Failed to process assembled data: ${error.message}`);
                }
            }

            return result;
        } catch (error) {
            this.logger.error(`Error processing chunked data:`, error.message);
            throw error;
        }
    }

    /**
     * Get session status
     */
    async getSessionStatus(sessionId: string): Promise<ChunkSessionStatus> {
        try {
            return this.handleChunkSessionService.getSessionStatus(sessionId);
        } catch (error) {
            this.logger.error(`Error getting session status for ${sessionId}:`, error.message);
            throw error;
        }
    }

    /**
     * Get all active sessions
     */
    async getActiveSessions(): Promise<ChunkSessionStatus[]> {
        try {
            return this.handleChunkSessionService.getActiveSessions();
        } catch (error) {
            this.logger.error('Error getting active sessions:', error.message);
            throw error;
        }
    }

    /**
     * Manually clean up a session
     */
    async cleanupSession(sessionId: string): Promise<boolean> {
        try {
            return this.handleChunkSessionService.cleanupSession(sessionId);
        } catch (error) {
            this.logger.error(`Error cleaning up session ${sessionId}:`, error.message);
            return false;
        }
    }

    /**
     * Send chunked data to an endpoint (utility method for client-side usage)
     */
    async sendChunkedData(
        chunks: ChunkData[],
        sendFunction: (chunk: ChunkData) => Promise<any>,
        onProgress?: (progress: { current: number; total: number; percentage: number }) => void
    ): Promise<any[]> {
        const results: any[] = [];

        try {
            for (let i = 0; i < chunks.length; i++) {
                const chunk = chunks[i];

                this.logger.debug(`Sending chunk ${i + 1}/${chunks.length} for session ${chunk.sessionId}`);

                const result = await sendFunction(chunk);
                results.push(result);

                // Call progress callback if provided
                if (onProgress) {
                    onProgress({
                        current: i + 1,
                        total: chunks.length,
                        percentage: Math.round(((i + 1) / chunks.length) * 100),
                    });
                }

                // Small delay to prevent overwhelming the server
                if (i < chunks.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 10));
                }
            }

            this.logger.log(`Successfully sent all ${chunks.length} chunks`);
            return results;
        } catch (error) {
            this.logger.error('Error sending chunked data:', error.message);
            throw error;
        }
    }

    /**
     * Estimate chunk requirements for given data
     */
    estimateChunkRequirements(data: any, chunkSize?: number): {
        estimatedSize: number;
        estimatedChunks: number;
        recommendedChunkSize: number;
    } {
        try {
            const serialized = JSON.stringify(data);
            const estimatedSize = Buffer.byteLength(serialized, 'utf8');
            const actualChunkSize = chunkSize || this.DEFAULT_CHUNK_SIZE;
            const estimatedChunks = Math.ceil(estimatedSize / actualChunkSize);

            // Recommend chunk size based on data size
            let recommendedChunkSize = this.DEFAULT_CHUNK_SIZE;
            if (estimatedSize < 10 * 1024) { // < 10KB
                recommendedChunkSize = estimatedSize; // Send in one chunk
            } else if (estimatedSize < 100 * 1024) { // < 100KB
                recommendedChunkSize = 50 * 1024; // 50KB chunks
            } else if (estimatedSize > 10 * 1024 * 1024) { // > 10MB
                recommendedChunkSize = 2 * 1024 * 1024; // 2MB chunks
            }

            return {
                estimatedSize,
                estimatedChunks,
                recommendedChunkSize,
            };
        } catch (error) {
            this.logger.error('Error estimating chunk requirements:', error.message);
            throw error;
        }
    }

    /**
     * Get a specific prepared chunk from a session
     */
    getPreparedChunk(sessionId: string, chunkIndex: number): any {
        try {
            return this.handleChunkSessionService.getPreparedChunk(sessionId, chunkIndex);
        } catch (error) {
            this.logger.error(`Error retrieving prepared chunk ${chunkIndex} from session ${sessionId}:`, error.message);
            throw error;
        }
    }
}
