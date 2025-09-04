import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, AllSessionsInitResponseDto, SessionChunkDto } from 'src/dto/racing-session.dto';
import { RacingSession } from 'src/schemas/racing-session.schema';
import * as crypto from 'crypto';

@Injectable()
export class RacingSessionService {
    constructor(@InjectModel(RacingSession.name) private racingSession: Model<RacingSession>) {
    }

    //
    async retrieveAllRacingSessionsBasicInfo(mapName: string, userId: string): Promise<SessionBasicInfoListDto | null> {

        try {
            let racingMap: SessionBasicInfoListDto = new SessionBasicInfoListDto();

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
                session.data = data.data;
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
    async createRacingSession(session_name: string, map: string, car_name: string, userId: string, data: any[]) {
        return this.racingSession.create({
            session_name: session_name,
            map: map,
            car_name: car_name,
            user_id: userId,
            data: data
        });
    }

    /**
     * Retrieves metadata for all racing sessions with chunking information
     * @param trackName - Track name to filter sessions
     * @param carName - Car name to filter sessions  
     * @param chunkSize - Size of each data chunk (default: 1000)
     * @returns Session metadata with chunking info
     */
    async initializeSessionsDownload(trackName: string, carName: string, chunkSize: number = 1000): Promise<AllSessionsInitResponseDto> {
        try {

            const sessions = await this.racingSession.find({ 'map': trackName, 'car_name': carName })
                .exec();

            const sessionMetadata = sessions.map(session => {
                const dataSize = session.data ? session.data.length : 0;
                const chunkCount = Math.ceil(dataSize / chunkSize);

                return {
                    sessionId: session._id.toString(),
                    session_name: session.session_name,
                    map: session.map,
                    car_name: session.car_name,
                    userId: session.user_id,
                    dataSize,
                    chunkCount
                };
            });

            const totalChunks = sessionMetadata.reduce((total, session) => total + session.chunkCount, 0);

            return {
                downloadId: crypto.randomUUID(),
                totalSessions: sessions.length,
                totalChunks,
                sessionMetadata
            };
        } catch (error) {
            throw new Error(`Failed to initialize sessions download: ${error.message}`);
        }
    }

    /**
     * Retrieves a specific chunk of session data
     * @param sessionId - The session ID
     * @param chunkIndex - The chunk index to retrieve
     * @param chunkSize - Size of each chunk (default: 1000)
     * @returns Session chunk data
     */
    async getSessionChunk(sessionId: string, chunkIndex: number, chunkSize: number = 1000): Promise<SessionChunkDto> {
        try {
            const session = await this.racingSession.findById(sessionId)
                .select('data')
                .exec();

            if (!session) {
                throw new Error('Session not found');
            }

            const data = session.data || [];
            const startIndex = chunkIndex * chunkSize;
            const endIndex = Math.min(startIndex + chunkSize, data.length);
            const chunkData = data.slice(startIndex, endIndex);
            const totalChunks = Math.ceil(data.length / chunkSize);

            return {
                downloadId: '', // Will be set by controller
                sessionId,
                chunkIndex,
                totalChunks,
                data: chunkData,
                isComplete: chunkIndex === totalChunks - 1
            };
        } catch (error) {
            throw new Error(`Failed to retrieve session chunk: ${error.message}`);
        }
    }

}
