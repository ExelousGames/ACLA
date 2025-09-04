import { Controller, Post, Get, Body, Param, Query } from '@nestjs/common';
import { ChunkClientService, ChunkData } from '../../shared/chunk-service';

/**
 * Example controller demonstrating how to use the ChunkClientService
 * for handling chunked JSON requests and responses
 */
@Controller('example-chunked')
export class ExampleChunkedController {
    constructor(
        private readonly chunkService: ChunkClientService
    ) { }

    /**
     * Example: Receive chunked data
     * POST /example-chunked/receive-chunks
     */
    @Post('receive-chunks')
    async receiveChunkedData(@Body() chunkData: ChunkData) {
        // Define what to do when all chunks are received and assembled
        const processCallback = async (assembledData: any) => {
            // Your business logic here - process the complete data
            console.log('Processing assembled data:', Object.keys(assembledData));

            // Example: Save to database, perform calculations, etc.
            return {
                processed: true,
                itemCount: Array.isArray(assembledData) ? assembledData.length : 1,
                timestamp: new Date(),
            };
        };

        // Use the chunk service to handle the incoming chunk
        return this.chunkService.handleIncomingChunk(chunkData, processCallback);
    }

    /**
     * Example: Prepare large data for chunked sending
     * POST /example-chunked/prepare-large-data
     */
    @Post('prepare-large-data')
    async prepareLargeData(
        @Body() requestData: { data: any; chunkSize?: number }
    ) {
        return this.chunkService.prepareDataForChunkedSending(
            requestData.data,
            requestData.chunkSize
        );
    }

    /**
     * Example: Get upload status
     * GET /example-chunked/status/:sessionId
     */
    @Get('status/:sessionId')
    async getUploadStatus(@Param('sessionId') sessionId: string) {
        return this.chunkService.getSessionStatus(sessionId);
    }

    /**
     * Example: Get all active sessions (for monitoring)
     * GET /example-chunked/active-sessions
     */
    @Get('active-sessions')
    async getActiveSessions() {
        return this.chunkService.getAllActiveSessions();
    }

    /**
     * Example: Clean up a session
     * POST /example-chunked/cleanup/:sessionId
     */
    @Post('cleanup/:sessionId')
    async cleanupSession(@Param('sessionId') sessionId: string) {
        return this.chunkService.cleanupSession(sessionId);
    }

    /**
     * Example: Estimate chunk requirements
     * POST /example-chunked/estimate
     */
    @Post('estimate')
    async estimateChunks(
        @Body() requestData: { data: any; chunkSize?: number }
    ) {
        return this.chunkService.estimateChunkRequirements(
            requestData.data,
            requestData.chunkSize
        );
    }

    /**
     * Example: Send data to another service in chunks
     * POST /example-chunked/send-to-service
     */
    @Post('send-to-service')
    async sendToAnotherService(
        @Body() requestData: {
            data: any;
            targetEndpoint: string;
            chunkSize?: number;
        }
    ) {
        // Define how to send each chunk (this would typically be an HTTP call)
        const sendFunction = async (chunk: ChunkData) => {
            // Example: Make HTTP request to target endpoint
            console.log(`Sending chunk ${chunk.chunkIndex + 1}/${chunk.totalChunks} to ${requestData.targetEndpoint}`);

            // In real implementation, you would use HttpService or similar:
            // return this.httpService.post(requestData.targetEndpoint, chunk).toPromise();

            // For demo, just return a mock response
            return {
                success: true,
                chunkIndex: chunk.chunkIndex,
                received: true,
            };
        };

        // Progress callback (optional)
        const onProgress = (progress: { current: number; total: number; percentage: number }) => {
            console.log(`Progress: ${progress.current}/${progress.total} (${progress.percentage}%)`);
        };

        return this.chunkService.sendDataInChunks(
            requestData.data,
            sendFunction,
            {
                chunkSize: requestData.chunkSize,
                onProgress,
            }
        );
    }
}
