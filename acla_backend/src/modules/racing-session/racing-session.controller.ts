import { Controller, Get, UseGuards, Request, Post, Body, Query, BadRequestException, Inject, forwardRef } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, UploadReacingSessionInitDto } from 'src/dto/racing-session.dto';
import { AiModelResponseDto } from 'src/dto/ai-model.dto';
import { RacingSessionService } from './racing-session.service';
import { AiModelService } from '../ai-model/ai-model.service';
import { UserInfoService } from '../user-info/user-info.service';
import { SessionAIModel } from 'src/schemas/session-ai-model.schema';
import { AiServiceClient, ModelsConfig } from '../ai-model/ai-service.client';
import { model } from 'mongoose';

@Controller('racing-session')
export class RacingSessionController {
    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        session_data_chunks: string[][];
        received: number;
    }>();

    constructor(
        private racingSessionService: RacingSessionService,
        @Inject(forwardRef(() => AiModelService))
        private aiModelService: AiModelService,
        private aiServiceClient: AiServiceClient,
        private userInfoService: UserInfoService
    ) { }

    @UseGuards(AuthGuard('jwt'))
    @Post('sessionbasiclist')
    retrieveAllRacingBasicSessionsInfo(@Request() req, @Body() body): Promise<SessionBasicInfoListDto | null> {
        return this.racingSessionService.retrieveAllRacingSessionsInfo(body.map_name, body.username);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('detailedSessionInfo')
    retrieveSessionDetailedInfo(@Request() req, @Body() body): Promise<RacingSessionDetailedInfoDto | null> {

        return this.racingSessionService.retrieveSessionDetailedInfo(body.id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload/init')
    async initUpload(@Body() metadata: UploadReacingSessionInitDto) {
        const uploadId = crypto.randomUUID();
        this.uploadStates.set(uploadId, {
            metadata,
            session_data_chunks: [],
            received: 0
        });
        return { uploadId: uploadId };
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload/chunk')
    async receiveChunk(
        @Body() body: { chunk: string[]; chunkIndex: number },
        @Query('uploadId') uploadId: string
    ) {
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist');
        }
        upload.session_data_chunks[body.chunkIndex] = body.chunk;
        upload.received++;


        return { receivedChunks: upload.session_data_chunks.length };
    }

    @Post('upload/complete')
    async completeUpload(
        @Body() completionData: any,
        @Query('uploadId') uploadId: string //Extracts values from the URL query string (the part after ? in a URL)
    ) {

        //get the data about the session upload
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist');
        }

        const fullDataset = upload.session_data_chunks.flat();

        console.log(`Upload complete for ID: ${uploadId}, total chunks: ${upload.session_data_chunks.length}, total records: ${fullDataset.length}`);

        // Create racing session in database
        const createdSession = await this.racingSessionService.createRacingSession(
            upload.metadata.sessionName,
            uploadId,
            upload.metadata.mapName,
            upload.metadata.userEmail,
            fullDataset
        );

        let modelReady = false;
        try {
            // First, find the user by email to get their ObjectId
            const userInfo = await this.userInfoService.findOne(upload.metadata.userEmail);

            if (!userInfo) {
                console.log('User not found for email:', upload.metadata.userEmail);
            } else {

                const userId = (userInfo as any).id.toString();
                // Check for active AI model first, before processing the session

                const modelsConfig: ModelsConfig[] = [
                    { config_id: "lap_prediction", target_variable: "lap_time", model_type: "lap_time_prediction", preferred_algorithm: "random_forest" }
                ];

                let activeModel: SessionAIModel | null = null;

                for (const modelConfig of modelsConfig) {

                    // Check if user has an active model for this track
                    activeModel = await this.aiModelService.findActiveModel(
                        userId,
                        upload.metadata.mapName,
                        modelConfig.model_type
                    );

                    // add active model if any
                    modelConfig.existing_model_data = activeModel ? activeModel : null;
                }

                // Train the models using the AI service client
                await this.aiServiceClient.trainModels({
                    session_id: createdSession.id,
                    telemetry_data: fullDataset,
                    models_config: modelsConfig,
                    user_id: userId,
                    parallel_training: false
                });

            }
        } catch (modelError) {
            console.error('Model setup failed:', modelError);
            modelReady = false;
        }

        return {
            message: 'Upload completed successfully',
            sessionId: uploadId,
            aiAnalysisAvailable: true
        };
    }
}
