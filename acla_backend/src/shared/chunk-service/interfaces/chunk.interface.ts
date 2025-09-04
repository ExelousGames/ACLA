export interface ChunkData {
    sessionId: string;
    chunkIndex: number;
    totalChunks: number;
    data: any;
    metadata?: {
        timestamp: Date;
        size: number;
        checksum?: string;
    };
}

export interface ChunkSession {
    sessionId: string;
    totalChunks: number;
    receivedChunks: number;
    chunks: Map<number, any>;
    createdAt: Date;
    lastUpdated: Date;
    status: 'pending' | 'complete' | 'failed';
    metadata?: any;
}

export interface ChunkPrepareOptions {
    data: any;
    chunkSize?: number;
    metadata?: any;
}

export interface ChunkPrepareResult {
    sessionId: string;
    totalChunks: number;
    chunks: ChunkData[];
}

export interface ChunkProcessResult {
    response: {
        success: boolean;
        message: string;
        sessionId: string;
        isComplete: boolean;
        receivedChunks: number;
        totalChunks: number;
    };
    processedResult?: any;
}

export interface ChunkSessionStatus {
    sessionId: string;
    status: 'pending' | 'complete' | 'failed';
    totalChunks: number;
    receivedChunks: number;
    completionPercentage: number;
    createdAt: Date;
    lastUpdated: Date;
    estimatedTimeRemaining?: number;
}
