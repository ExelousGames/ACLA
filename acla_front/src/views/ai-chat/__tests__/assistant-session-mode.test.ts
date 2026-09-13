import {
    resolveAssistantRecordedSessionId,
    resolveRegisteredAssistantIdentity,
} from '../assistant-session-mode';

describe('assistant session mode resolution', () => {
    const createRegistration = (overrides: Record<string, unknown> = {}) => ({
        screenId: 'live-session',
        assistantMode: 'live' as const,
        label: 'Live Session',
        componentRef: { current: null },
        ...overrides,
    });

    it('uses the Front Desk fallback while registration is temporarily unavailable', () => {
        expect(resolveRegisteredAssistantIdentity(null)).toEqual({
            sessionMode: 'front_desk',
            sessionId: undefined,
            label: 'Front Desk',
            title: 'AI Assistant - Front Desk',
        });
    });

    it('uses the active registration instead of recording or dashboard state', () => {
        expect(resolveRegisteredAssistantIdentity(createRegistration())).toMatchObject({
            sessionMode: 'live',
            label: 'Live Session',
        });
    });

    it('uses the registered recorded id for title and conversation identity', () => {
        expect(resolveRegisteredAssistantIdentity(createRegistration({
            screenId: 'recorded-session',
            assistantMode: 'recorded',
            label: 'Race 12',
            recordedSessionId: 'session-1',
        }) as any)).toMatchObject({
            sessionMode: 'recorded',
            sessionId: 'session-1',
            label: 'Race 12',
            title: 'AI Assistant - Race 12',
        });
    });

    it.each(['live', 'front_desk', 'user_summary'] as const)(
        'does not expose a recorded session id in %s mode',
        (sessionMode) => {
            expect(resolveAssistantRecordedSessionId(sessionMode, 'session-1')).toBeUndefined();
        },
    );

    it('keeps the recorded session id in recorded mode', () => {
        expect(resolveAssistantRecordedSessionId('recorded', 'session-1')).toBe('session-1');
    });

});
