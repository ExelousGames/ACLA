import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { AnalysisSessionMetadataDto, RacingSessionDetailedInfoDto, SessionBasicInfoListDto, AllSessionsInitResponseDto, SessionChunkDto, MapBasicInfoListDto } from 'src/dto/racing-session.dto';
import { RacingSession } from 'src/schemas/racing-session.schema';
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';
import { ObjectId } from 'mongodb';
import { Types } from 'mongoose';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { GameRecordedFrom } from 'src/racing-session-game';

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
            let sessionList: SessionBasicInfoListDto = new SessionBasicInfoListDto();
            //find all sessions with the map name and user id, only return session_name and _id
            const data = await this.racingSession.find({ 'map': mapName, 'user_id': userId }).select('session_name user_id').exec();
            data.forEach((element) => {
                sessionList.list.push({
                    name: element.session_name,
                    sessionId: element._id.toString()
                });
            });
            return sessionList;

        }
        catch (e) {
            // Handle errors appropriately
            throw new Error(`Failed to process data: ${e.message}`);
        }

    }

    async retrieveAllSessionMapBasicInfo(userId: string): Promise<MapBasicInfoListDto | null> {
        try {
            const filter = userId ? { user_id: userId } : {};
            const mapNames = await this.racingSession.distinct('map', filter).exec();
            const result = new MapBasicInfoListDto();
            result.list = mapNames
                .filter((name): name is string => typeof name === 'string' && name.length > 0)
                .sort((a, b) => a.localeCompare(b))
                .map((name) => ({ name }));

            return result;
        } catch (e) {
            throw new Error(`Failed to process data: ${e.message}`);
        }
    }

    async retrieveSessionDetailedInfo(id: string): Promise<RacingSessionDetailedInfoDto | null> {
        try {
            let session: RacingSessionDetailedInfoDto = new RacingSessionDetailedInfoDto;
            const data = await this.racingSession.findOne({ 'user_id': id }).exec();

            if (data) {
                session.session_name = data.session_name;
                session.game_recorded_from = data.game_recorded_from;
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
        gameRecordedFrom: GameRecordedFrom,
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
            game_recorded_from: gameRecordedFrom,
            dataChunkFileIds: dataChunkFileIds,
            chunkSize: chunkSize,
            totalChunks: totalChunks,
            totalDataPoints: data.length,
            created_date: new Date()
        });
    }

    /**
     * Uploads a single chunk of session data to GridFS.
     * @param chunk - The data chunk to upload.
     * @param metadata - Metadata for the chunk.
     * @returns The ObjectId of the uploaded file.
     */
    async uploadSessionChunk(
        chunk: any[],
        metadata: {
            session_name: string;
            map: string;
            car_name: string;
            userId: string;
            chunkIndex: number;
            chunkSize: number;
        }
    ): Promise<ObjectId> {
        const filename = `session_${metadata.session_name}_${metadata.map}_${metadata.car_name}_chunk_${metadata.chunkIndex}_${Date.now()}.json`;
        const fileId = await this.gridfsService.uploadJSON(
            chunk,
            filename,
            {
                ...metadata,
                createdAt: new Date()
            },
            GRIDFS_BUCKETS.RACING_SESSIONS
        );
        return fileId as unknown as ObjectId;
    }

    /**
     * Creates a racing session from a list of pre-uploaded GridFS file IDs.
     * @param session_name 
     * @param map 
     * @param car_name 
     * @param userId 
     * @param dataChunkFileIds 
     * @param totalDataPoints 
     * @param chunkSize 
     * @returns 
     */
    async createRacingSessionFromChunks(
        session_name: string,
        map: string,
        car_name: string,
        userId: string,
        gameRecordedFrom: GameRecordedFrom,
        dataChunkFileIds: ObjectId[],
        totalDataPoints: number,
        chunkSize: number
    ) {
        return this.racingSession.create({
            session_name,
            map,
            car_name,
            user_id: userId,
            game_recorded_from: gameRecordedFrom,
            dataChunkFileIds: dataChunkFileIds,
            chunkSize: chunkSize,
            totalChunks: dataChunkFileIds.length,
            totalDataPoints: totalDataPoints,
            created_date: new Date()
        });
    }

    /**
     * Initializes a streaming download by returning metadata only.
     * Actual telemetry rows are streamed from GridFS by download/chunk.
     */
    async initializeSessionsDownload(trackName?: string, carName?: string, chunkSize: number = 1000, sessionId?: string): Promise<AllSessionsInitResponseDto> {
        try {
            const sessions = await this.listSessionsForDownload(trackName, carName, sessionId);

            const sessionMetadata = sessions.map(session => ({
                sessionId: session._id.toString(),
                session_name: session.session_name,
                game_recorded_from: session.game_recorded_from,
                map: session.map,
                car_name: session.car_name,
                userId: session.user_id,
                dataSize: session.totalDataPoints || 0,
                dataPoints: session.totalDataPoints || 0,
                chunkCount: session.dataChunkFileIds?.length || session.totalChunks || 0
            }));

            return {
                downloadId: crypto.randomUUID(),
                totalSessions: sessions.length,
                totalChunks: sessionMetadata.reduce((total, session) => total + (session.chunkCount || 0), 0),
                sessionMetadata
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

    async listSessionsForDownload(trackName?: string, carName?: string, sessionId?: string): Promise<any[]> {
        const filter: any = {};
        if (sessionId) filter._id = sessionId;
        if (trackName) filter.map = trackName;
        if (carName) filter.car_name = carName;

        return this.racingSession.find(filter)
            .select('session_name game_recorded_from map car_name user_id totalDataPoints totalChunks dataChunkFileIds')
            .exec();
    }

    async getSessionDownloadChunk(sessionId: string, chunkIndex: number): Promise<{
        stream: NodeJS.ReadableStream;
        fileSize: number;
        totalChunks: number;
        dataPoints: number;
    }> {
        if (!Types.ObjectId.isValid(sessionId)) {
            throw new Error('Invalid session id');
        }

        const session = await this.racingSession.findById(sessionId)
            .select('dataChunkFileIds totalDataPoints')
            .exec();

        if (!session) {
            throw new Error('Session not found');
        }

        const fileIds = session.dataChunkFileIds || [];
        if (chunkIndex < 0 || chunkIndex >= fileIds.length) {
            throw new Error('Chunk index out of range');
        }

        const fileId = new ObjectId(fileIds[chunkIndex].toString());
        const [stream, fileSize] = await Promise.all([
            this.gridfsService.downloadJSONStream(fileId, GRIDFS_BUCKETS.RACING_SESSIONS),
            this.gridfsService.getFileSize(fileId, GRIDFS_BUCKETS.RACING_SESSIONS),
        ]);

        return {
            stream,
            fileSize,
            totalChunks: fileIds.length,
            dataPoints: session.totalDataPoints || 0,
        };
    }

    async listUserSessionsForAnalysis(userId: string, sessionLimit = 10): Promise<AnalysisSessionMetadataDto[]> {
        const limit = Math.max(1, Math.min(Math.floor(Number(sessionLimit) || 10), 10));
        const sessions = await this.racingSession.find({ user_id: userId })
            .select('session_name game_recorded_from map car_name user_id totalDataPoints totalChunks chunkSize dataChunkFileIds')
            .sort({ created_date: -1, _id: -1 })
            .limit(limit)
            .exec();

        return sessions.map((session) => ({
            sessionId: session._id.toString(),
            session_name: session.session_name,
            game_recorded_from: session.game_recorded_from,
            map: session.map,
            car_name: session.car_name,
            userId: session.user_id,
            totalDataPoints: session.totalDataPoints || 0,
            totalChunks: session.dataChunkFileIds?.length || session.totalChunks || 0,
            chunkSize: session.chunkSize || 0,
        }));
    }

    async getUserSessionAnalysisChunk(userId: string, sessionId: string, chunkIndex: number): Promise<{
        stream: NodeJS.ReadableStream;
        fileSize: number;
        totalChunks: number;
    }> {
        if (!Types.ObjectId.isValid(sessionId)) {
            throw new Error('Invalid session id');
        }

        const session = await this.racingSession.findOne({ _id: sessionId, user_id: userId })
            .select('dataChunkFileIds totalChunks')
            .exec();

        if (!session) {
            throw new Error('Session not found');
        }

        const fileIds = session.dataChunkFileIds || [];
        if (chunkIndex < 0 || chunkIndex >= fileIds.length) {
            throw new Error('Chunk index out of range');
        }

        const fileId = new ObjectId(fileIds[chunkIndex].toString());
        const [stream, fileSize] = await Promise.all([
            this.gridfsService.downloadJSONStream(fileId, GRIDFS_BUCKETS.RACING_SESSIONS),
            this.gridfsService.getFileSize(fileId, GRIDFS_BUCKETS.RACING_SESSIONS),
        ]);

        return {
            stream,
            fileSize,
            totalChunks: fileIds.length,
        };
    }

    async getSessionTelemetryForClassification(userId: string, sessionId: string): Promise<{
        sessionId: string;
        trackName: string;
        carName: string;
        telemetryData: any[];
    }> {
        if (!Types.ObjectId.isValid(sessionId)) {
            throw new Error('Invalid session id');
        }

        const session = await this.racingSession.findById(sessionId)
            .select('map car_name user_id dataChunkFileIds')
            .exec();

        if (!session) {
            throw new Error('Session not found');
        }

        if (session.user_id !== userId) {
            throw new Error('Session not found or access denied');
        }

        const fileIds = session.dataChunkFileIds || [];
        if (fileIds.length === 0) {
            throw new Error('Session has no telemetry chunks');
        }

        const telemetryData: any[] = [];
        for (const fileId of fileIds) {
            const chunk = await this.gridfsService.downloadJSON(
                new ObjectId(fileId.toString()),
                GRIDFS_BUCKETS.RACING_SESSIONS,
            );

            if (!Array.isArray(chunk)) {
                throw new Error('Session telemetry chunk is not an array');
            }

            telemetryData.push(...chunk);
        }

        if (telemetryData.length === 0) {
            throw new Error('Session has no telemetry rows');
        }

        return {
            sessionId,
            trackName: session.map || '',
            carName: session.car_name || '',
            telemetryData,
        };
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
