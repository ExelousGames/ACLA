import React, {
    MutableRefObject,
    createContext,
    useCallback,
    useContext,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import type { ToolHandlerContext } from 'views/lap-analysis/ai-chat/use-voice-conversation';

export type AiChatAssistantMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';

export type AiChatJsonPrimitive = string | number | boolean | null;
export type AiChatJsonValue =
    | AiChatJsonPrimitive
    | AiChatJsonValue[]
    | { [key: string]: AiChatJsonValue };

export const toAiChatJsonValue = (value: unknown): AiChatJsonValue => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (Array.isArray(value)) return value.map(toAiChatJsonValue);
    if (value && typeof value === 'object') {
        return Object.fromEntries(
            Object.entries(value).flatMap(([key, item]) => (
                typeof item === 'undefined' || typeof item === 'function' || typeof item === 'symbol'
                    ? []
                    : [[key, toAiChatJsonValue(item)]]
            )),
        );
    }
    return null;
};

export const toAiChatJsonRecord = (
    value: Record<string, unknown>,
): Record<string, AiChatJsonValue> => toAiChatJsonValue(value) as Record<string, AiChatJsonValue>;

export type AiChatScreenStatusTone = 'neutral' | 'info' | 'success' | 'warning' | 'error';

export interface AiChatScreenPillInfo {
    title: string;
    description: string;
    status: {
        label: string;
        tone: AiChatScreenStatusTone;
    };
    facts: Array<{
        label: string;
        value: string;
    }>;
}

export interface AiChatScreenToolHandlerContext extends ToolHandlerContext {
    executeCore: (name: string, args: Record<string, any>) => Promise<any>;
}

export type AiChatScreenToolHandler = (
    args: Record<string, any>,
    context: AiChatScreenToolHandlerContext,
) => Promise<any>;

export type AiChatScreenToolHandlers = Record<string, AiChatScreenToolHandler>;

export interface AiChatScreenHandle {
    getAiContext: () => Record<string, AiChatJsonValue>;
    getToolHandlers: () => AiChatScreenToolHandlers;
}

export interface AiChatScreenRegistration {
    screenId: string;
    assistantMode: AiChatAssistantMode;
    pillLabel: string;
    recordedSessionId?: string;
    getPillInfo: () => AiChatScreenPillInfo;
    componentRef: MutableRefObject<AiChatScreenHandle | null>;
}

export const LIVE_SCREEN_TOOL_NAMES = [
    'query_telemetry_metric',
    '_get_telemetry_for_scope',
    'get_event_log',
    'get_next_corner',
    'get_live_focus_section',
    'get_live_section_history',
    'set_live_range_todo_list',
    'update_live_range_todo_list',
    'get_live_range_todo_list',
    'analyze_live_recorded_analysis',
    '_get_live_section_telemetry',
    '_record_live_section_classification',
] as const;

export const RECORDED_SCREEN_TOOL_NAMES = [
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
] as const;

export const USER_SUMMARY_SCREEN_TOOL_NAMES = [
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
] as const;

export const SCREEN_VISUALIZATION_TOOL_NAMES = [
    'get_visualization_capabilities',
    'show_map',
    'open_visualization_chart',
    'close_visualization_chart',
    'invoke_visualization_control',
    'update_guidance_once',
    'add_imitation_guidance_chart',
    'remove_imitation_guidance_chart',
    'disable_ui_component',
] as const;

export const SCREEN_OWNED_TOOL_NAMES = new Set<string>([
    ...LIVE_SCREEN_TOOL_NAMES,
    ...RECORDED_SCREEN_TOOL_NAMES,
    ...USER_SUMMARY_SCREEN_TOOL_NAMES,
    ...SCREEN_VISUALIZATION_TOOL_NAMES,
]);

export const createAiChatScreenToolHandlers = (
    names: readonly string[],
): AiChatScreenToolHandlers => Object.fromEntries(names.map((name) => [
    name,
    async (args: Record<string, any>, context: AiChatScreenToolHandlerContext) => (
        context.executeCore(name, args)
    ),
]));

type RegistrationOwner = symbol;

interface AiChatScreenContextValue {
    activeScreen: AiChatScreenRegistration | null;
    registerScreen: (owner: RegistrationOwner, registration: AiChatScreenRegistration) => void;
    unregisterScreen: (owner: RegistrationOwner) => void;
}

const AiChatScreenContext = createContext<AiChatScreenContextValue>({
    activeScreen: null,
    registerScreen: () => undefined,
    unregisterScreen: () => undefined,
});

export const AiChatScreenProvider = ({ children }: { children: React.ReactNode }) => {
    const [activeScreen, setActiveScreen] = useState<AiChatScreenRegistration | null>(null);
    const activeOwnerRef = useRef<RegistrationOwner | null>(null);

    const registerScreen = useCallback((
        owner: RegistrationOwner,
        registration: AiChatScreenRegistration,
    ) => {
        activeOwnerRef.current = owner;
        setActiveScreen((current) => current === registration ? current : registration);
    }, []);

    const unregisterScreen = useCallback((owner: RegistrationOwner) => {
        if (activeOwnerRef.current !== owner) return;
        activeOwnerRef.current = null;
        setActiveScreen(null);
    }, []);

    const value = useMemo(() => ({
        activeScreen,
        registerScreen,
        unregisterScreen,
    }), [activeScreen, registerScreen, unregisterScreen]);

    return (
        <AiChatScreenContext.Provider value={value}>
            {children}
        </AiChatScreenContext.Provider>
    );
};

export const useAiChatScreen = () => useContext(AiChatScreenContext);

export const useAiChatScreenRegistration = (
    registration: AiChatScreenRegistration,
) => {
    const { registerScreen, unregisterScreen } = useAiChatScreen();
    const ownerRef = useRef<RegistrationOwner>(Symbol('ai-chat-screen-owner'));

    useLayoutEffect(() => {
        registerScreen(ownerRef.current, registration);
    }, [registerScreen, registration]);

    useLayoutEffect(() => () => {
        unregisterScreen(ownerRef.current);
    }, [unregisterScreen]);
};
