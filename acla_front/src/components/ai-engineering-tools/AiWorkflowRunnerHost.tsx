import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useAiToolComponentRefs } from 'contexts/AiToolComponentRefContext';
import { AiToolComponentBase } from './AiToolComponentBase';
import { GoalDisplay, GoalRunner, type GoalSnapshot } from './Goal';
import {
    LiveRangeTodoListDisplay,
    LiveRangeTodoListRunner,
} from './LiveRangeTodoList';
import type { LiveRangeTodoListSnapshot } from './live-range-todo-list-types';
import ProcedurePlan, {
    ProcedurePlanRunner,
    type ProcedurePlanSnapshot,
} from './ProcedurePlan';

export type AiWorkflowRunnerSnapshot =
    | { kind: 'goal'; runner: GoalRunner; snapshot: GoalSnapshot | null }
    | { kind: 'live_range_todo'; runner: LiveRangeTodoListRunner; snapshot: LiveRangeTodoListSnapshot | null }
    | { kind: 'procedure_plan'; runner: ProcedurePlanRunner; snapshot: ProcedurePlanSnapshot | null }
    | null;

export type AiWorkflowRunnerHostProps = {
    surface?: 'chat' | 'pill';
    telemetry?: Record<string, any> | null;
    onSnapshotChange?: (active: AiWorkflowRunnerSnapshot) => void;
};

const getActiveSnapshot = (
    runner: AiToolComponentBase<unknown> | null,
): AiWorkflowRunnerSnapshot => {
    if (runner instanceof GoalRunner) {
        return { kind: 'goal', runner, snapshot: runner.getSnapshot() };
    }
    if (runner instanceof LiveRangeTodoListRunner) {
        return { kind: 'live_range_todo', runner, snapshot: runner.getSnapshot() };
    }
    if (runner instanceof ProcedurePlanRunner) {
        return { kind: 'procedure_plan', runner, snapshot: runner.getSnapshot() };
    }
    return null;
};

export const AiWorkflowRunnerHost: React.FC<AiWorkflowRunnerHostProps> = ({
    surface = 'chat',
    telemetry,
    onSnapshotChange,
}) => {
    const { directory, revision } = useAiToolComponentRefs();
    const runner = directory.findBaseComponentRef()?.current ?? null;
    const [active, setActive] = useState<AiWorkflowRunnerSnapshot>(() => (
        getActiveSnapshot(runner)
    ));
    const latestRunnerRef = useRef<AiToolComponentBase<unknown> | null>(runner);
    latestRunnerRef.current = runner;

    useLayoutEffect(() => {
        const publish = () => {
            const next = getActiveSnapshot(runner);
            setActive(next);
            onSnapshotChange?.(next);
        };
        publish();
        return runner?.subscribe(publish);
    }, [onSnapshotChange, runner]);

    useEffect(() => {
        if (runner instanceof LiveRangeTodoListRunner) {
            runner.acceptTelemetry(telemetry);
        }
    }, [runner, telemetry]);

    useEffect(() => () => {
        latestRunnerRef.current?.dispose();
    }, []);

    // The directory revision deliberately participates in the render even
    // though the stable runner reference is read above.
    void revision;

    if (!active?.snapshot) return null;
    if (active.kind === 'goal') {
        return <GoalDisplay snapshot={active.snapshot} surface={surface} />;
    }
    if (active.kind === 'live_range_todo') {
        return <LiveRangeTodoListDisplay snapshot={active.snapshot} surface={surface} />;
    }
    return (
        <ProcedurePlan
            plan={active.snapshot}
            surface={surface}
            onClear={() => active.runner.clear()}
        />
    );
};

export default AiWorkflowRunnerHost;
