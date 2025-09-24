import { Injectable, InternalServerErrorException, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { InjectConnection } from '@nestjs/mongoose';
import { Connection } from 'mongoose';
import { GridFSBucket, ObjectId } from 'mongodb';
import { Readable } from 'stream';

// Constants for GridFS bucket names
export const GRIDFS_BUCKETS = {
    AI_MODELS: 'ai_models',
    RACING_SESSIONS: 'racing_sessions',
} as const;

@Injectable()
export class GridFSService implements OnModuleInit, OnModuleDestroy {
    private gridFSBuckets: Map<string, GridFSBucket> = new Map();
    private isInitialized = false;

    constructor(@InjectConnection() private connection: Connection) { }

    async onModuleInit() {
        await this.initializeGridFS();
    }

    async onModuleDestroy() {
        console.log('GridFS service shutting down...');
        this.isInitialized = false;
        this.gridFSBuckets.clear();
    }

    private async initializeGridFS(): Promise<void> {
        try {
            // Wait for connection to be ready with better retry logic
            if (this.connection.readyState !== 1) {
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => {
                        reject(new Error('Database connection timeout after 30 seconds'));
                    }, 30000);

                    // Handle different connection events
                    const cleanup = () => {
                        clearTimeout(timeout);
                        this.connection.removeListener('connected', onConnected);
                        this.connection.removeListener('error', onError);
                    };

                    const onConnected = () => {
                        cleanup();
                        resolve(void 0);
                    };

                    const onError = (error: Error) => {
                        cleanup();
                        reject(new Error(`Database connection error: ${error.message}`));
                    };

                    this.connection.once('connected', onConnected);
                    this.connection.once('error', onError);

                    // Check if already connected
                    if (this.connection.readyState === 1) {
                        cleanup();
                        resolve(void 0);
                    }
                });
            }

            if (!this.connection.db) {
                throw new Error('Database connection not established after connection ready');
            }

            // Initialize default buckets
            this.initializeBucket(GRIDFS_BUCKETS.AI_MODELS);
            this.initializeBucket(GRIDFS_BUCKETS.RACING_SESSIONS);

            this.isInitialized = true;
            console.log('GridFS initialized successfully');
        } catch (error) {
            console.error('Failed to initialize GridFS:', error);
            this.isInitialized = false;
            throw new InternalServerErrorException(`Failed to initialize GridFS: ${error.message}`);
        }
    }

    private initializeBucket(bucketName: string): GridFSBucket {
        if (this.gridFSBuckets.has(bucketName)) {
            return this.gridFSBuckets.get(bucketName)!;
        }

        if (!this.connection.db) {
            throw new InternalServerErrorException('Database connection not established');
        }

        const bucket = new GridFSBucket(this.connection.db, { bucketName });
        this.gridFSBuckets.set(bucketName, bucket);
        console.log(`GridFS bucket '${bucketName}' initialized`);
        return bucket;
    }

    private async getBucket(bucketName: string = GRIDFS_BUCKETS.AI_MODELS): Promise<GridFSBucket> {
        await this.ensureInitialized();

        if (!this.gridFSBuckets.has(bucketName)) {
            return this.initializeBucket(bucketName);
        }
        return this.gridFSBuckets.get(bucketName)!;
    }

    private async ensureInitialized(): Promise<void> {
        if (!this.isInitialized) {
            // Add retry logic for Docker environments where services start in parallel
            let retries = 3;
            let lastError: Error | null = null;

            while (retries > 0 && !this.isInitialized) {
                try {
                    await this.initializeGridFS();
                    break;
                } catch (error) {
                    lastError = error as Error;
                    retries--;
                    if (retries > 0) {
                        console.log(`GridFS initialization failed, retrying... (${retries} attempts remaining)`);
                        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds before retry
                    }
                }
            }

            if (!this.isInitialized && lastError) {
                throw lastError;
            }
        }
    }

    async uploadFile(buffer: Buffer, filename: string, metadata?: any, bucketName?: string): Promise<ObjectId> {
        const bucket = await this.getBucket(bucketName);

        try {
            const uploadStream = bucket.openUploadStream(filename, {
                metadata: {
                    ...metadata,
                    uploadedAt: new Date(),
                    bucketName: bucketName || GRIDFS_BUCKETS.AI_MODELS
                }
            });

            const readable = new Readable();
            readable.push(buffer);
            readable.push(null);

            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Upload timeout after 60 seconds'));
                }, 60000);

                readable.pipe(uploadStream)
                    .on('error', (error) => {
                        clearTimeout(timeout);
                        reject(error);
                    })
                    .on('finish', () => {
                        clearTimeout(timeout);
                        resolve(uploadStream.id as ObjectId);
                    });
            });
        } catch (error) {
            throw new InternalServerErrorException(`Failed to upload file to ${bucketName || GRIDFS_BUCKETS.AI_MODELS}: ${error.message}`);
        }
    }

    /**
     * Upload a stream to GridFS without buffering the entire content in memory.
     */
    async uploadStream(readable: Readable, filename: string, metadata?: any, bucketName?: string): Promise<ObjectId> {
        const bucket = await this.getBucket(bucketName);

        try {
            const uploadStream = bucket.openUploadStream(filename, {
                metadata: {
                    ...metadata,
                    uploadedAt: new Date(),
                    bucketName: bucketName || GRIDFS_BUCKETS.AI_MODELS
                }
            });

            return new Promise<ObjectId>((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Upload timeout after 60 seconds'));
                }, 60000);

                readable
                    .on('error', (err) => {
                        clearTimeout(timeout);
                        reject(err);
                    })
                    .pipe(uploadStream)
                    .on('error', (error) => {
                        clearTimeout(timeout);
                        reject(error);
                    })
                    .on('finish', () => {
                        clearTimeout(timeout);
                        resolve(uploadStream.id as ObjectId);
                    });
            });
        } catch (error) {
            throw new InternalServerErrorException(`Failed to upload stream to ${bucketName || GRIDFS_BUCKETS.AI_MODELS}: ${error.message}`);
        }
    }

    /**
     * Upload a file to GridFS.
     * @param data File data as a Buffer.
     * @param filename Name of the file.
     * @param metadata Metadata to associate with the file.
     * @param bucketName Name of the GridFS bucket.
     * @returns The ID of the uploaded file.
     */
    async uploadJSON(data: any, filename: string, metadata?: any, bucketName?: string): Promise<ObjectId> {
        const buffer = Buffer.from(JSON.stringify(data));
        return this.uploadFile(buffer, filename, metadata, bucketName);
    }

    async downloadJSON(fileId: ObjectId, bucketName?: string): Promise<any> {
        const buffer = await this.downloadFile(fileId, bucketName);
        return JSON.parse(buffer.toString());
    }

    /**
     * Stream download for large JSON files to avoid memory issues.
     * This method streams the file and parses JSON in chunks if needed.
     */
    async downloadJSONStream(fileId: ObjectId, bucketName?: string): Promise<NodeJS.ReadableStream> {
        const bucket = await this.getBucket(bucketName);

        try {
            return bucket.openDownloadStream(fileId);
        } catch (error) {
            throw new InternalServerErrorException(`Failed to create download stream from ${bucketName || GRIDFS_BUCKETS.AI_MODELS}: ${error.message}`);
        }
    }

    /**
     * Check file size before download to determine if streaming is needed
     */
    async getFileSize(fileId: ObjectId, bucketName?: string): Promise<number> {
        const fileInfo = await this.getFileInfo(fileId, bucketName);
        return fileInfo?.length || 0;
    }

    async downloadFile(fileId: ObjectId, bucketName?: string): Promise<Buffer> {
        const bucket = await this.getBucket(bucketName);

        try {
            const chunks: Buffer[] = [];
            const downloadStream = bucket.openDownloadStream(fileId);

            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Download timeout after 60 seconds'));
                }, 60000);

                downloadStream
                    .on('data', (chunk) => chunks.push(chunk))
                    .on('error', (error) => {
                        clearTimeout(timeout);
                        reject(new Error(`Failed to download file from ${bucketName || GRIDFS_BUCKETS.AI_MODELS}: ${error.message}`));
                    })
                    .on('end', () => {
                        clearTimeout(timeout);
                        resolve(Buffer.concat(chunks));
                    });
            });
        } catch (error) {
            throw new InternalServerErrorException(`Failed to download file from ${bucketName || GRIDFS_BUCKETS.AI_MODELS}: ${error.message}`);
        }
    }

    async deleteFile(fileId: ObjectId, bucketName?: string): Promise<void> {
        const bucket = await this.getBucket(bucketName);

        try {
            await bucket.delete(fileId);
        } catch (error) {
            throw new InternalServerErrorException(`Failed to delete file: ${error.message}`);
        }
    }

    async getFileInfo(fileId: ObjectId, bucketName?: string): Promise<any> {
        const bucket = await this.getBucket(bucketName);

        try {
            const cursor = bucket.find({ _id: fileId });
            const files = await cursor.toArray();
            return files.length > 0 ? files[0] : null;
        } catch (error) {
            throw new InternalServerErrorException(`Failed to get file info: ${error.message}`);
        }
    }

    async findFiles(filter: any = {}, bucketName?: string): Promise<any[]> {
        const bucket = await this.getBucket(bucketName);

        try {
            const cursor = bucket.find(filter);
            return await cursor.toArray();
        } catch (error) {
            throw new InternalServerErrorException(`Failed to find files: ${error.message}`);
        }
    }

    // Health check method for Docker with comprehensive checks
    async isHealthy(bucketName?: string): Promise<boolean> {
        try {
            // Check if service is initialized
            if (!this.isInitialized) {
                console.warn('GridFS health check: Service not initialized');
                return false;
            }

            // Check database connection
            if (this.connection.readyState !== 1) {
                console.warn('GridFS health check: Database not connected');
                return false;
            }

            const bucket = await this.getBucket(bucketName);

            // Try to list files to verify GridFS is working
            await bucket.find({}).limit(1).toArray();

            // Verify collections exist
            const targetBucket = bucketName || GRIDFS_BUCKETS.AI_MODELS;
            const collections = await this.connection.db!.listCollections().toArray();
            const hasFilesCollection = collections.some(col => col.name === `${targetBucket}.files`);

            console.log(`GridFS health check passed for bucket: ${targetBucket}`);
            return true;
        } catch (error) {
            console.error(`GridFS health check failed for bucket ${bucketName || GRIDFS_BUCKETS.AI_MODELS}:`, error.message);
            return false;
        }
    }

    // Get GridFS statistics for a specific bucket
    async getStats(bucketName: string = GRIDFS_BUCKETS.AI_MODELS): Promise<any> {
        await this.ensureInitialized();

        try {
            if (!this.connection.db) {
                throw new Error('Database connection not available');
            }

            const filesCollection = `${bucketName}.files`;
            const chunksCollection = `${bucketName}.chunks`;

            const fileCount = await this.connection.db.collection(filesCollection).countDocuments();
            const chunkCount = await this.connection.db.collection(chunksCollection).countDocuments();

            // Get total size by aggregating file lengths
            const sizeAggregation = await this.connection.db.collection(filesCollection).aggregate([
                { $group: { _id: null, totalSize: { $sum: '$length' } } }
            ]).toArray();

            const totalSize = sizeAggregation.length > 0 ? sizeAggregation[0].totalSize : 0;

            return {
                bucketName,
                fileCount,
                chunkCount,
                totalSize,
                avgObjSize: fileCount > 0 ? Math.round(totalSize / fileCount) : 0
            };
        } catch (error) {
            console.error('Failed to get GridFS stats:', error);
            return {
                bucketName,
                fileCount: 0,
                chunkCount: 0,
                totalSize: 0,
                avgObjSize: 0
            };
        }
    }

    // List all available buckets
    listBuckets(): string[] {
        return Array.from(this.gridFSBuckets.keys());
    }

    // Get stats for all buckets
    async getAllBucketsStats(): Promise<any[]> {
        const buckets = this.listBuckets();
        const stats: any[] = [];

        for (const bucket of buckets) {
            stats.push(await this.getStats(bucket));
        }

        return stats;
    }

    /**
     * Download a specific byte range from a GridFS file
     * This is useful for chunked transfer of large files without loading the entire file
     * @param fileId ObjectId of the file in GridFS
     * @param bucketName Name of the GridFS bucket
     * @param startByte Starting byte position (inclusive)
     * @param endByte Ending byte position (inclusive)
     * @returns String content of the specified byte range
     */
    async downloadFileRange(fileId: ObjectId, bucketName: string, startByte: number, endByte: number): Promise<string> {
        const bucket = await this.getBucket(bucketName);

        try {
            // Create a download stream with start and end options
            const downloadStream = bucket.openDownloadStream(fileId, {
                start: startByte,
                end: endByte
            });

            const chunks: Buffer[] = [];

            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Download range timeout after 30 seconds for bytes ${startByte}-${endByte}`));
                }, 30000);

                downloadStream.on('data', (chunk: Buffer) => {
                    chunks.push(chunk);
                });

                downloadStream.on('end', () => {
                    clearTimeout(timeout);
                    const buffer = Buffer.concat(chunks);
                    resolve(buffer.toString('utf8'));
                });

                downloadStream.on('error', (error) => {
                    clearTimeout(timeout);
                    console.error(`GridFS download range error for bytes ${startByte}-${endByte}:`, error);
                    reject(new InternalServerErrorException(`Failed to download range from ${bucketName}: ${error.message}`));
                });
            });
        } catch (error) {
            throw new InternalServerErrorException(`Failed to create download range stream from ${bucketName}: ${error.message}`);
        }
    }

    /**
     * Download a specific byte range from a GridFS file as base64 string
     * This is useful for chunked transfer of binary data over HTTP
     * @param fileId ObjectId of the file in GridFS
     * @param bucketName Name of the GridFS bucket
     * @param startByte Starting byte position (inclusive)
     * @param endByte Ending byte position (inclusive)
     * @returns Base64 encoded content of the specified byte range
     */
    async downloadFileRangeAsBase64(fileId: ObjectId, bucketName: string, startByte: number, endByte: number): Promise<string> {
        const bucket = await this.getBucket(bucketName);

        try {
            // Create a download stream with start and end options
            const downloadStream = bucket.openDownloadStream(fileId, {
                start: startByte,
                end: endByte
            });

            const chunks: Buffer[] = [];

            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Download range timeout after 30 seconds for bytes ${startByte}-${endByte}`));
                }, 30000);

                downloadStream.on('data', (chunk: Buffer) => {
                    chunks.push(chunk);
                });

                downloadStream.on('end', () => {
                    clearTimeout(timeout);
                    const buffer = Buffer.concat(chunks);
                    resolve(buffer.toString('base64'));
                });

                downloadStream.on('error', (error) => {
                    clearTimeout(timeout);
                    console.error(`GridFS download range error for bytes ${startByte}-${endByte}:`, error);
                    reject(new InternalServerErrorException(`Failed to download range from ${bucketName}: ${error.message}`));
                });
            });
        } catch (error) {
            throw new InternalServerErrorException(`Failed to create download range stream from ${bucketName}: ${error.message}`);
        }
    }

    // Get detailed connection and service information for debugging
    async getServiceInfo(): Promise<any> {
        return {
            initialized: this.isInitialized,
            connectionState: this.connection.readyState,
            connectionStates: {
                0: 'disconnected',
                1: 'connected',
                2: 'connecting',
                3: 'disconnecting'
            },
            currentState: this.connection.readyState === 1 ? 'connected' :
                this.connection.readyState === 0 ? 'disconnected' :
                    this.connection.readyState === 2 ? 'connecting' : 'disconnecting',
            availableBuckets: this.listBuckets(),
            dbName: this.connection.db?.databaseName || 'unknown',
            host: this.connection.host || 'unknown',
            port: this.connection.port || 'unknown'
        };
    }

    // Reinitialize GridFS (useful for Docker container restarts)
    async reinitialize(): Promise<void> {
        console.log('Reinitializing GridFS...');
        this.isInitialized = false;
        this.gridFSBuckets.clear();
        await this.initializeGridFS();
    }

    // Get bucket type from bucket name (useful for type checking)
    static getBucketType(bucketName: string): keyof typeof GRIDFS_BUCKETS | null {
        const entry = Object.entries(GRIDFS_BUCKETS).find(([_, value]) => value === bucketName);
        return entry ? entry[0] as keyof typeof GRIDFS_BUCKETS : null;
    }

    // Check if bucket name is valid
    static isValidBucketName(bucketName: string): boolean {
        return Object.values(GRIDFS_BUCKETS).includes(bucketName as any);
    }

    // Get all available bucket names
    static getAllBucketNames(): string[] {
        return Object.values(GRIDFS_BUCKETS);
    }
}
