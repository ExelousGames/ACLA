import { Module, Global } from '@nestjs/common';
import { ChunkClientService } from './chunk.service';
import { ChunkHandlerService } from './chunk-handler.service';
import { HandleChunkSessionService } from './handle-chunk-session.service';

/**
 * Global chunk service module
 * Provides chunked JSON request/response handling capabilities
 * to all controllers in the application
 */
@Global()
@Module({
  controllers: [],
  providers: [
    HandleChunkSessionService,
    ChunkHandlerService,
    ChunkClientService,
  ],
  exports: [
    ChunkClientService,
    ChunkHandlerService,
    HandleChunkSessionService,
  ],
})
export class ChunkModule { }
