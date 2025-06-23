import { Controller, Get, UseGuards, Request, Post, Body, Query, BadRequestException } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { SessionBasicInfoListDto, UploadReacingSessionInitDto } from 'src/dto/racing-session.dto';
import { RacingSessionService } from './racing-session.service';

@Controller('racing-session')
export class RacingSessionController {
    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        session_data_chunks: string[][];
        received: number;
    }>();

    constructor(private racingSessionService: RacingSessionService) { }

    @UseGuards(AuthGuard('jwt'))
    @Get('allmapbasicinfos')
    retrieveAllRacingBasicSessionsInfo(@Request() req): SessionBasicInfoListDto {
        this.racingSessionService.retrieveAllRacingSessionsInfo();
        return req.user;
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload/init')
    async initUpload(@Body() metadata: UploadReacingSessionInitDto) {
        console.log("here");
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
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist');
        }

        const fullDataset = upload.session_data_chunks.flat();

        this.racingSessionService.createRacingSession(upload.metadata.sessionName, uploadId, upload.metadata.mapName, upload.metadata.userEmail, fullDataset);

        this.uploadStates.delete(uploadId);

    }
}
