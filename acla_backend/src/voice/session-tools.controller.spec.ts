import {
    BadRequestException,
    ForbiddenException,
    RequestMethod,
} from '@nestjs/common';
import {
    GUARDS_METADATA,
    METHOD_METADATA,
    PATH_METADATA,
} from '@nestjs/common/constants';
import { SessionToolsController } from './session-tools.controller';

describe('SessionToolsController', () => {
    const controller = new SessionToolsController();
    const originalUsername = process.env.AI_SERVICE_USERNAME;

    beforeEach(() => {
        process.env.AI_SERVICE_USERNAME = 'ai-service@example.com';
    });

    afterAll(() => {
        if (originalUsername === undefined) {
            delete process.env.AI_SERVICE_USERNAME;
        } else {
            process.env.AI_SERVICE_USERNAME = originalUsername;
        }
    });

    const request = (username = 'ai-service@example.com') => ({
        user: { username },
    });

    it('requires the JWT auth guard', () => {
        const guards = Reflect.getMetadata(
            GUARDS_METADATA,
            SessionToolsController.prototype.getSessionTools,
        );
        expect(guards).toHaveLength(1);
    });

    it('exposes POST /session-tools', () => {
        expect(Reflect.getMetadata(
            PATH_METADATA,
            SessionToolsController.prototype.getSessionTools,
        )).toBe('session-tools');
        expect(Reflect.getMetadata(
            METHOD_METADATA,
            SessionToolsController.prototype.getSessionTools,
        )).toBe(RequestMethod.POST);
    });

    it('restricts the authenticated endpoint to the configured AI service account', () => {
        expect(() => controller.getSessionTools(
            request('driver@example.com'),
            { session_context: { session_mode: 'live' } },
        )).toThrow(ForbiddenException);

        delete process.env.AI_SERVICE_USERNAME;
        expect(() => controller.getSessionTools(
            request(),
            { session_context: { session_mode: 'live' } },
        )).toThrow(ForbiddenException);
    });

    it.each([
        undefined,
        {},
        { session_context: {} },
        { session_context: { session_mode: 'unknown' } },
        { session_context: { session_mode: 'live', agent_mode: 'unknown' } },
    ])('rejects invalid session context: %p', (body) => {
        expect(() => controller.getSessionTools(request(), body))
            .toThrow(BadRequestException);
    });

    it('preserves the recorded-session allowlist', () => {
        const tools = controller.getSessionTools(request(), {
            session_context: { session_mode: 'recorded' },
        });
        const names = tools.map(({ name }) => name);

        expect(names).toEqual(expect.arrayContaining([
            'run_recorded_ai_analysis',
            'get_recorded_session_analysis',
            'stop_agent_session',
        ]));
        expect(names).not.toEqual(expect.arrayContaining([
            'start_agent_session',
            'classify_live_section',
            'restart_live_baseline',
        ]));
    });

    it('preserves the live analyst allowlist and exact response shape', () => {
        const tools = controller.getSessionTools(request(), {
            session_context: {
                session_mode: 'live',
                agent_mode: 'live_performance_analyst',
            },
        });
        const names = tools.map(({ name }) => name);

        expect(names).toEqual(expect.arrayContaining([
            'collect_live_baseline',
            'get_live_analysis_mistake_count',
            'create_goal',
            'retry_goal_task',
            'classify_live_section',
        ]));
        expect(names).not.toContain('start_agent_session');
        expect(tools.every((tool) => (
            Object.keys(tool).sort().join(',')
            === 'description,name,properties,required'
        ))).toBe(true);
        expect(tools.every(({ description }) => typeof description === 'string'))
            .toBe(true);
        expect(tools.some((tool) => 'title' in tool)).toBe(false);
    });

    it.each(['front_desk', 'live', 'recorded', 'user_summary'])(
        'accepts session mode %s',
        (sessionMode) => {
            expect(controller.getSessionTools(request(), {
                session_context: { session_mode: sessionMode },
            }).length).toBeGreaterThan(0);
        },
    );
});
