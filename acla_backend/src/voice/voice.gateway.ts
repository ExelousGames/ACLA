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
 * proxy (binary PCM audio + JSON tool relay frames). It sanitizes compact
 * session context on control frames, then pipes frames in both directions.
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
        const chatLlmModel = this.normalizeChatLlmModel(
            parsed.searchParams.get('chat_llm_model'),
        );
        const chatSessionAction = parsed.searchParams.get('chat_session_action');
        const chatSessionId = parsed.searchParams.get('chat_session_id');

        this.bridge(
            client,
            userId,
            sessionId,
            chatLlmModel,
            chatSessionAction,
            chatSessionId,
        );
    }

    private aiServiceWsBase(): string {
        const httpUrl = new URL(this.aiServiceUrl);
        const proto = httpUrl.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${httpUrl.host}`;
    }

    private normalizeChatLlmModel(model: string | null): string | null {
        const normalized = (model || '').trim();
        return normalized || null;
    }

    private buildUpstreamUrl(
        userId: string,
        sessionId: string,
        chatLlmModel: string | null = null,
        chatSessionAction: string | null = null,
        chatSessionId: string | null = null,
    ): string {
        const params = new URLSearchParams();
        params.set('user_id', userId);
        if (sessionId) params.set('session_id', sessionId);
        if (chatLlmModel) params.set('chat_llm_model', chatLlmModel);
        if (chatSessionAction) params.set('chat_session_action', chatSessionAction);
        if (chatSessionId) params.set('chat_session_id', chatSessionId);
        return `${this.aiServiceWsBase()}/voice/stream?${params.toString()}`;
    }

    private sanitizeSessionContext(value: unknown): Record<string, unknown> {
        if (!value || typeof value !== 'object' || Array.isArray(value)) {
            return {};
        }

        const sessionContext = value as Record<string, unknown>;
        return {
            ...(typeof sessionContext.session_mode === 'string'
                ? { session_mode: sessionContext.session_mode }
                : {}),
            ...(typeof sessionContext.agent_mode === 'string'
                ? { agent_mode: sessionContext.agent_mode }
                : {}),
        };
    }

    private sanitizeContextFrame(data: RawData, isBinary: boolean): { data: RawData | string; isBinary: boolean } {
        if (isBinary) {
            return { data, isBinary };
        }

        const text = typeof data === 'string' ? data : data.toString();
        let payload: any;
        try {
            payload = JSON.parse(text);
        } catch {
            return { data, isBinary };
        }

        if (!payload || typeof payload !== 'object') {
            return { data, isBinary };
        }

        const isContextFrame = (
            payload.type === 'session_info'
            || payload.type === 'session_context'
            || payload.type === 'user_text'
        );
        if (!isContextFrame) {
            return { data, isBinary };
        }

        const {
            session_mode: _sessionMode,
            agent_mode: _agentMode,
            context_kind: _contextKind,
            active_agent_session: _activeAgentSession,
            agent_session: _agentSession,
            agent_modes: _agentModes,
            tools: _tools,
            tool_metadata: _toolMetadata,
            query_scope_schema: _queryScopeSchema,
            tool_result_handling: _toolResultHandling,
            ...contextFrame
        } = payload;
        const sanitizedPayload = {
            ...contextFrame,
            session_context: this.sanitizeSessionContext(payload.session_context),
        };

        return {
            data: JSON.stringify(sanitizedPayload),
            isBinary: false,
        };
    }

    private bridge(
        client: WsClient,
        userId: string,
        sessionId: string,
        chatLlmModel: string | null,
        chatSessionAction: string | null,
        chatSessionId: string | null,
    ): void {
        const upstreamUrl = this.buildUpstreamUrl(
            userId,
            sessionId,
            chatLlmModel,
            chatSessionAction,
            chatSessionId,
        );

        const upstream = new WsClient(upstreamUrl);

        // Hold client → upstream messages until upstream finishes opening —
        // the browser audio worklet starts pushing PCM frames immediately.
        const queue: Array<{ data: RawData | string; isBinary: boolean }> = [];
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
            const next = this.sanitizeContextFrame(data, isBinary);
            if (!upstreamOpen) {
                queue.push(next);
                return;
            }
            if (upstream.readyState === WsClient.OPEN) {
                upstream.send(next.data, { binary: next.isBinary });
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
