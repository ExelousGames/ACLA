import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, AllSessionsInitResponseDto, SessionChunkDto } from 'src/dto/racing-session.dto';
import { RacingSession } from 'src/schemas/racing-session.schema';
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';
import { ObjectId } from 'mongodb';
import { Types } from 'mongoose';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { pipeline, Readable } from 'stream';
import { promisify } from 'util';

@Injectable()
export class RacingSessionService {
    constructor(
        @InjectModel(RacingSession.name) private racingSession: Model<RacingSession>,
        private readonly gridfsService: GridFSService,
    ) { }

    /**
     * Large telemetry datasets are stored exclusively in GridFS as chunked JSON files.
     * Stored metadata per session document:
     *  - dataChunkFileIds: ordered GridFS file IDs (JSON arrays of telemetry rows)
     *  - chunkSize: size used for splitting when uploaded
     *  - totalChunks: number of chunks
     *  - totalDataPoints: total number of telemetry rows
     * Public API surfaces (upload/init, upload/chunk, upload/complete, download/init, download/chunk) are unchanged.
     */

    /**
     * Retrieves basic information about all racing sessions for a specific map and user.
     * @param mapName - The name of the racing map.
     * @param userId - The ID of the user.
     * @returns A promise that resolves to a list of basic session information.
     */
    async retrieveAllRacingSessionsBasicInfo(mapName: string, userId: string): Promise<SessionBasicInfoListDto | null> {

        try {
            let racingMap: SessionBasicInfoListDto = new SessionBasicInfoListDto();
            //find all sessions with the map name and user id, only return session_name and _id
            const data = await this.racingSession.find({ 'map': mapName, 'user_id': userId }).select('session_name user_id').exec();
            data.forEach((element) => {
                racingMap.list.push({
                    name: element.session_name,
                    sessionId: element._id.toString()
                });
            });
            return racingMap;

        }
        catch (e) {
            // Handle errors appropriately
            throw new Error(`Failed to process data: ${e.message}`);
        }

    }

    async retrieveSessionDetailedInfo(id: string): Promise<RacingSessionDetailedInfoDto | null> {
        try {
            let session: RacingSessionDetailedInfoDto = new RacingSessionDetailedInfoDto;
            const data = await this.racingSession.findOne({ 'user_id': id }).exec();

            if (data) {
                session.session_name = data.session_name;
                session.map = data.map;
                session.userId = data.user_id.toString();
                session.points = data.points;
                // Telemetry data is stored in GridFS chunks; detailed endpoint returns empty array placeholder
                session.data = [];
            }

            return session;
        } catch (error) {
            // Handle errors appropriately
            throw new Error(`Failed to process data: ${error.message}`);
        };
    }


    /**
     * Creates a new racing session.
     * @param session_name 
     * @param map 
     * @param car_name 
     * @param userId 
     * @param data 
     * @returns 
     */
    async createRacingSession(
        session_name: string,
        map: string,
        car_name: string,
        userId: string,
        data: any[],
        options?: { chunkSize?: number }
    ) {
        const chunkSize = options?.chunkSize || 1000;
        const dataChunkFileIds: ObjectId[] = [];
        const totalChunks = Math.ceil(data.length / chunkSize);
        for (let i = 0; i < totalChunks; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, data.length);
            const chunk = data.slice(start, end);
            const filename = `session_${session_name}_${map}_${car_name}_chunk_${i}_${Date.now()}.json`;
            const fileId = await this.gridfsService.uploadJSON(
                chunk,
                filename,
                {
                    session_name,
                    map,
                    car_name,
                    userId,
                    chunkIndex: i,
                    totalChunks,
                    chunkSize,
                    createdAt: new Date()
                },
                GRIDFS_BUCKETS.RACING_SESSIONS
            );
            dataChunkFileIds.push(fileId as unknown as ObjectId);
        }
        return this.racingSession.create({
            session_name,
            map,
            car_name,
            user_id: userId,
            dataChunkFileIds: dataChunkFileIds,
            chunkSize: chunkSize,
            totalChunks: totalChunks,
            totalDataPoints: data.length,
            created_date: new Date()
        });
    }

    /**
     * Initializes streaming download by preparing session files on disk
     * @param trackName - Track name to filter sessions (optional)
     * @param carName - Car name to filter sessions (optional)
     * @param chunkSize - Size of each data chunk (legacy parameter, ignored)
     * @returns Session metadata with file streaming information plus streaming context
     */
    async initializeSessionsDownload(trackName?: string, carName?: string, chunkSize: number = 1000): Promise<AllSessionsInitResponseDto & { streamingContext?: any }> {
        try {
            // Prepare sessions for streaming - this creates temporary files
            const streamingData = await this.prepareSessionsForStreaming(trackName, carName);

            // Map to the expected DTO format
            const sessionMetadata = streamingData.sessionFiles.map(sessionFile => ({
                sessionId: sessionFile.sessionId,
                session_name: sessionFile.session_name,
                map: sessionFile.map,
                car_name: sessionFile.car_name,
                userId: sessionFile.userId,
                dataSize: sessionFile.dataPoints,
                fileSize: sessionFile.fileSize,
                dataPoints: sessionFile.dataPoints,
                // Legacy fields for backward compatibility
                chunkCount: 1 // Each session is now a single file
            }));

            return {
                downloadId: streamingData.downloadId,
                totalSessions: streamingData.totalSessions,
                totalChunks: streamingData.totalSessions, // Each session is one "chunk" now
                sessionMetadata,
                streamingContext: {
                    sessionFiles: streamingData.sessionFiles,
                    tempDir: streamingData.tempDir
                }
            };
        } catch (error) {
            throw new Error(`Failed to initialize sessions download: ${error.message}`);
        }
    }

    /**
     * Retrieves streaming information for a specific session
     * @param sessionId - The session ID
     * @param chunkIndex - The chunk index (legacy parameter, ignored in streaming mode)
     * @param chunkSize - Size of each chunk (legacy parameter, ignored)
     * @returns Session streaming information
     */
    async getSessionChunk(sessionId: string, chunkIndex: number, chunkSize: number = 1000): Promise<SessionChunkDto> {
        try {
            // Find the session to verify it exists
            const session = await this.racingSession.findById(sessionId)
                .select('session_name totalDataPoints')
                .exec();

            if (!session) {
                throw new Error('Session not found');
            }

            // For streaming, we don't load the data into memory
            // Instead, we return metadata that the controller will use for streaming
            return {
                downloadId: '', // Will be set by controller
                sessionId,
                filePath: '', // Will be set by controller based on download state
                fileSize: 0, // Will be set by controller
                contentType: 'application/json',
                dataPoints: session.totalDataPoints || 0,
                // Legacy fields for backward compatibility
                chunkIndex: 0,
                totalChunks: 1,
                data: [], // Empty array for backward compatibility
                isComplete: true
            };
        } catch (error) {
            throw new Error(`Failed to retrieve session chunk: ${error.message}`);
        }
    }

    /**
     * Prepares session data as temporary files for streaming without memory buffering
     * Combines all chunks for each session into a single temporary file
     * @param trackName - Track name filter (optional)
     * @param carName - Car name filter (optional)
     * @returns Session metadata with temporary file paths for streaming
     */
    async prepareSessionsForStreaming(trackName?: string, carName?: string): Promise<{
        downloadId: string;
        totalSessions: number;
        sessionFiles: Array<{
            sessionId: string;
            session_name: string;
            map: string;
            car_name: string;
            userId: string;
            filePath: string;
            fileSize: number;
            dataPoints: number;
        }>;
        tempDir: string;
    }> {
        const pipelineAsync = promisify(pipeline);

        try {
            // Build filter
            const filter: any = {};
            if (trackName) filter.map = trackName;
            if (carName) filter.car_name = carName;

            const sessions = await this.racingSession.find(filter).exec();

            // Create temporary directory for this download session
            const downloadId = crypto.randomUUID();
            const tempDir = path.resolve(process.cwd(), 'session_recording', 'temp', 'streaming', downloadId);

            // Ensure temp directory exists
            await fs.promises.mkdir(tempDir, { recursive: true });

            const sessionFiles: Array<{
                sessionId: string;
                session_name: string;
                map: string;
                car_name: string;
                userId: string;
                filePath: string;
                fileSize: number;
                dataPoints: number;
            }> = [];

            // Process each session to create temporary files
            for (const session of sessions) {
                const sessionFilePath = path.join(tempDir, `${session._id.toString()}.json`);
                const writeStream = fs.createWriteStream(sessionFilePath);

                let totalDataPoints = 0;
                let isFirstChunk = true;

                // Start JSON object with _id and data array
                writeStream.write(`{"_id":"${session._id.toString()}","data":[`);

                if (session.dataChunkFileIds && session.dataChunkFileIds.length > 0) {
                    // Stream each chunk and append to file
                    for (let i = 0; i < session.dataChunkFileIds.length; i++) {
                        const fileId = session.dataChunkFileIds[i];

                        try {
                            // Get readable stream from GridFS
                            const readStream = await this.gridfsService.downloadJSONStream(new ObjectId(fileId.toString()), GRIDFS_BUCKETS.RACING_SESSIONS);

                            // Parse JSON chunk and write to file
                            let chunkData = '';

                            await new Promise<void>((resolve, reject) => {
                                readStream.on('data', (chunk: Buffer) => {
                                    chunkData += chunk.toString();
                                });

                                readStream.on('end', () => {
                                    try {
                                        const jsonData = JSON.parse(chunkData);
                                        const dataArray = Array.isArray(jsonData) ? jsonData : [];

                                        // Add comma separator between chunks (except for first chunk)
                                        if (!isFirstChunk && dataArray.length > 0) {
                                            writeStream.write(',');
                                        }

                                        // Write data points without array brackets
                                        if (dataArray.length > 0) {
                                            const dataString = JSON.stringify(dataArray).slice(1, -1); // Remove [ and ]
                                            writeStream.write(dataString);
                                            totalDataPoints += dataArray.length;
                                            isFirstChunk = false;
                                        }

                                        resolve();
                                    } catch (parseError) {
                                        reject(new Error(`Failed to parse chunk ${i} for session ${session._id}: ${parseError.message}`));
                                    }
                                });

                                readStream.on('error', reject);
                            });

                        } catch (chunkError) {
                            console.warn(`Failed to process chunk ${i} for session ${session._id}: ${chunkError.message}`);
                        }
                    }
                }

                // End JSON array and object, and close file
                writeStream.write(']}');
                writeStream.end();

                // Wait for write stream to finish
                await new Promise<void>((resolve, reject) => {
                    writeStream.on('finish', resolve);
                    writeStream.on('error', reject);
                });

                // Get file size
                const stats = await fs.promises.stat(sessionFilePath);

                sessionFiles.push({
                    sessionId: session._id.toString(),
                    session_name: session.session_name,
                    map: session.map,
                    car_name: session.car_name,
                    userId: session.user_id,
                    filePath: sessionFilePath,
                    fileSize: stats.size,
                    dataPoints: totalDataPoints
                });
            }

            return {
                downloadId,
                totalSessions: sessions.length,
                sessionFiles,
                tempDir
            };

        } catch (error) {
            throw new Error(`Failed to prepare sessions for streaming: ${error.message}`);
        }
    }

    /**
     * Clean up temporary streaming files for a specific download session
     * @param downloadId - The download session ID
     */
    async cleanupStreamingFiles(downloadId: string): Promise<void> {
        try {
            const tempDir = path.resolve(process.cwd(), 'session_recording', 'temp', 'streaming', downloadId);

            // Check if directory exists
            if (fs.existsSync(tempDir)) {
                // Remove all files in the directory
                const files = await fs.promises.readdir(tempDir);
                for (const file of files) {
                    await fs.promises.unlink(path.join(tempDir, file));
                }

                // Remove the directory
                await fs.promises.rmdir(tempDir);
            }
        } catch (error) {
            console.warn(`Failed to cleanup streaming files for ${downloadId}: ${error.message}`);
        }
    }

    /**
     * Clean up old streaming temporary files (older than 2 hours)
     */
    async cleanupOldStreamingFiles(): Promise<void> {
        try {
            const streamingDir = path.resolve(process.cwd(), 'session_recording', 'temp', 'streaming');

            if (!fs.existsSync(streamingDir)) {
                return;
            }

            const directories = await fs.promises.readdir(streamingDir);
            const twoHoursAgo = Date.now() - (2 * 60 * 60 * 1000);
            let cleanedCount = 0;

            for (const dir of directories) {
                try {
                    const dirPath = path.join(streamingDir, dir);
                    const stats = await fs.promises.stat(dirPath);

                    if (stats.isDirectory() && stats.mtime.getTime() < twoHoursAgo) {
                        // Remove all files in directory
                        const files = await fs.promises.readdir(dirPath);
                        for (const file of files) {
                            await fs.promises.unlink(path.join(dirPath, file));
                        }

                        // Remove directory
                        await fs.promises.rmdir(dirPath);
                        cleanedCount++;
                    }
                } catch (error) {
                    console.warn(`Could not process streaming directory ${dir}: ${error.message}`);
                }
            }

            if (cleanedCount > 0) {
                console.log(`Cleaned up ${cleanedCount} old streaming directories`);
            }
        } catch (error) {
            console.warn(`Could not clean up old streaming files: ${error.message}`);
        }
    }


}
