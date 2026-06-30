import { useCallback, useRef, useState } from 'react';
import {
    type BaselineCollectionTag,
    type BaselineLapRecord,
} from './BaselineCollectionTracker';
import {
    type ToolOutputEmitter,
    type ToolOutputEnvelope,
} from './ai-tool-base';

type BaselineCollectionRuntimeOptions = {
    onToolOutput?: ToolOutputEmitter;
};

export const useBaselineCollectionRuntime = ({
    onToolOutput,
}: BaselineCollectionRuntimeOptions = {}) => {
    const [enabled, setEnabled] = useState(false);
    const [tag, setTag] = useState<BaselineCollectionTag | null>(null);
    const [restartToken, setRestartToken] = useState(0);
    const tagRef = useRef<BaselineCollectionTag | null>(null);
    const lapRecordRef = useRef<BaselineLapRecord | null>(null);
    const toolOutputRef = useRef<ToolOutputEnvelope | null>(null);
    const toolOutputListenersRef = useRef<Set<ToolOutputEmitter>>(new Set());

    const handleTagChange = useCallback((nextTag: BaselineCollectionTag | null) => {
        tagRef.current = nextTag;
        setTag(nextTag);
    }, []);

    const handleLapRecordChange = useCallback((record: BaselineLapRecord | null) => {
        lapRecordRef.current = record;
    }, []);

    const handleToolOutput = useCallback<ToolOutputEmitter>((envelope, options) => {
        toolOutputRef.current = envelope;
        toolOutputListenersRef.current.forEach((listener) => {
            listener(envelope, options);
        });
        onToolOutput?.(envelope, options);
    }, [onToolOutput]);

    const getTag = useCallback(() => tagRef.current, []);
    const getLapRecord = useCallback(() => lapRecordRef.current, []);
    const getToolOutput = useCallback(() => toolOutputRef.current, []);

    const subscribeToolOutput = useCallback((listener: ToolOutputEmitter) => {
        toolOutputListenersRef.current.add(listener);
        return () => {
            toolOutputListenersRef.current.delete(listener);
        };
    }, []);

    const restart = useCallback(() => {
        tagRef.current = null;
        lapRecordRef.current = null;
        toolOutputRef.current = null;
        setTag(null);
        setRestartToken((current) => current + 1);
    }, []);

    return {
        enabled,
        setEnabled,
        tag,
        restartToken,
        restart,
        trackerProps: {
            enabled,
            restartToken,
            onTagChange: handleTagChange,
            onLapRecordChange: handleLapRecordChange,
            onToolOutput: handleToolOutput,
        },
        getTag,
        getLapRecord,
        getToolOutput,
        subscribeToolOutput,
    };
};
