import { Logger } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import {
    OnGatewayConnection,
    WebSocketGateway,
} from '@nestjs/websockets';
import { IncomingMessage } from 'http';
import { URL } from 'url';
import { WebSocket as WsClient, RawData } from 'ws';

/**
 * Voice WS gateway — backend edge for /voice/stream.
 *
 * Same auth model as the REST text path (`@Post('ai-query')` in
 * user-session-ai-model.controller): JWT is verified at the NestJS
 * boundary, `user_id` is derived from the verified claim, and the
 * connection is relayed to the AI service as a trusted inner hop. The
 * AI service stays auth-free on its private port.
 *
 * The gateway does NOT use `@SubscribeMessage` — this is a frame-level
 * passthrough proxy (binary PCM audio + JSON tool relay frames). We
 * subscribe to raw `message` events on the client and pipe them to
 * upstream unchanged, in both directions.
 */
@WebSocketGateway({ path: '/voice/stream' })
export class VoiceGateway implements OnGatewayConnection {
    private readonly logger = new Logger(VoiceGateway.name);
    private readonly aiServiceUrl =
        process.env.AI_SERVICE_URL || 'http://localhost:8000';

    constructor(private readonly jwt: JwtService) {}

    handleConnection(client: WsClient, req: IncomingMessage): void {
        const parsed = new URL(req.url || '', 'http://localhost');
        const token = parsed.searchParams.get('token');
        if (!token) {
            client.close(1008, 'Unauthorized');
            return;
        }

        let payload: any;
        try {
            payload = this.jwt.verify(token);
        } catch {
            client.close(1008, 'Unauthorized');
            return;
        }

        // Same claim shape as AuthService.giveJWTToken — payload.id is the
        // Mongo user _id. JwtStrategy.validate() reads the same field.
        const userId: string | undefined = payload?.id;
        if (!userId) {
            client.close(1008, 'Unauthorized');
            return;
        }

        // Client-supplied — forwarded as-is to the AI service, same as
        // the text path forwards `context`. Not used for authorization.
        const sessionId = parsed.searchParams.get('session_id') || '';

        this.bridge(client, userId, sessionId);
    }

    private aiServiceWsBase(): string {
        const httpUrl = new URL(this.aiServiceUrl);
        const proto = httpUrl.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${httpUrl.host}`;
    }

    private bridge(client: WsClient, userId: string, sessionId: string): void {
        const params = new URLSearchParams();
        params.set('user_id', userId);
        if (sessionId) params.set('session_id', sessionId);
        const upstreamUrl = `${this.aiServiceWsBase()}/voice/stream?${params.toString()}`;

        const upstream = new WsClient(upstreamUrl);

        // Hold client → upstream messages until upstream finishes opening —
        // the browser audio worklet starts pushing PCM frames immediately.
        const queue: Array<{ data: RawData; isBinary: boolean }> = [];
        let upstreamOpen = false;

        const closeBoth = (code?: number, reason?: string): void => {
            try { client.close(code, reason); } catch { /* ignore */ }
            try { upstream.close(code, reason); } catch { /* ignore */ }
        };

        upstream.on('open', () => {
            upstreamOpen = true;
            while (queue.length > 0) {
                const m = queue.shift()!;
                upstream.send(m.data, { binary: m.isBinary });
            }
        });

        upstream.on('message', (data, isBinary) => {
            if (client.readyState === WsClient.OPEN) {
                client.send(data, { binary: isBinary });
            }
        });

        upstream.on('close', (code, reason) =>
            closeBoth(code, reason.toString()),
        );
        upstream.on('error', (err) => {
            this.logger.warn(`upstream error: ${err.message}`);
            closeBoth(1011, 'upstream error');
        });

        client.on('message', (data, isBinary) => {
            if (!upstreamOpen) {
                queue.push({ data, isBinary });
                return;
            }
            if (upstream.readyState === WsClient.OPEN) {
                upstream.send(data, { binary: isBinary });
            }
        });

        client.on('close', (code, reason) =>
            closeBoth(code, reason.toString()),
        );
        client.on('error', (err) => {
            this.logger.warn(`client error: ${err.message}`);
            closeBoth(1011, 'client error');
        });
    }
}
