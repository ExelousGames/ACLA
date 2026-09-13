import {
    BadRequestException,
    Body,
    Controller,
    Post,
    UseGuards,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { getModelCommandsForSessionContext } from '../shared/ai/model-command-protocol-registry';

const SESSION_MODES = new Set([
    'front_desk',
    'live',
    'recorded',
    'user_summary',
]);
const AGENT_MODES = new Set([
    'track_guide',
    'overtake',
    'live_performance_analyst',
]);

@Controller()
export class ModelCommandProtocolController {
    @UseGuards(AuthGuard('jwt'))
    @Post('model-command-protocol')
    getModelCommands(@Body() body: unknown) {
        if (!body || typeof body !== 'object' || Array.isArray(body)) {
            throw new BadRequestException('session_context is required');
        }
        const sessionContext = (body as Record<string, unknown>).session_context;
        if (
            !sessionContext
            || typeof sessionContext !== 'object'
            || Array.isArray(sessionContext)
        ) {
            throw new BadRequestException('session_context must be an object');
        }

        const context = sessionContext as Record<string, unknown>;
        if (!SESSION_MODES.has(context.session_mode as string)) {
            throw new BadRequestException('session_context.session_mode is invalid');
        }
        if (
            context.agent_mode !== undefined
            && !AGENT_MODES.has(context.agent_mode as string)
        ) {
            throw new BadRequestException('session_context.agent_mode is invalid');
        }

        return getModelCommandsForSessionContext({
            session_mode: context.session_mode,
            ...(context.agent_mode === undefined
                ? {}
                : { agent_mode: context.agent_mode }),
        });
    }
}
