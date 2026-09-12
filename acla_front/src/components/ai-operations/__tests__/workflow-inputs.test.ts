import {
    asTool,
    asWorkflow,
    createControlledOperation,
    createOperation,
    LiveRangeTodoListRunner,
    parseProcedurePlanInput,
    parseRepeatablePlanInput,
    ProcedurePlanRunner,
    RepeatablePlanRunner,
    type ProcedurePlanInput,
    type RepeatablePlanInput,
    type ToolDispatcher,
    type LiveRangeTodoEventInput,
} from '..';

const procedure = (...names: string[]): ProcedurePlanInput => ({
    workflow: { name: 'set_procedure_plan',
        goal: 'Review telemetry',
        operations: names.map((name) => ({ operation: { name: name, title: name, arguments: {} } })) as unknown as ProcedurePlanInput['workflow']['operations'],
    },
});

const repeatable = (names = ['query_lap_analysis_result'], stop = 'query_lap_analysis_result'): RepeatablePlanInput => ({
    workflow: { name: 'create_repeatable_plan',
        goal: 'Improve consistency',
        operations: names.map((name, index) => ({ operation: { name: name, id: String(index), title: name } })) as unknown as RepeatablePlanInput['workflow']['operations'],
        stop_when: {
            tool: { name: stop,} as unknown as RepeatablePlanInput['workflow']['stop_when']['tool'],
            operator: 'eq',
            target: 0,
        },
    },
});

describe('strict workflow inputs', () => {
    it.each([
        { goal: 'Legacy', requests: [{ name: 'query_lap_analysis_result', title: 'Query', payload: {} }] },
        { workflow: { name: 'set_procedure_plan', goal: 'Legacy', tools: [{ tool: { name: 'query_lap_analysis_result', title: 'Query', arguments: {} } }] } },
        { workflow: { name: 'set_procedure_plan', goal: 'Review', operations: [{ tool: { name: 'query_lap_analysis_result', title: 'Query', arguments: {} } }] } },
        { workflow: { name: 'set_procedure_plan', operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query', arguments: {} } }] } },
        { workflow: { name: 'set_procedure_plan', goal: 'Review', operations: [] } },
        { workflow: { name: 'set_procedure_plan', goal: 'Review', operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query' } }] } },
        { workflow: { name: 'set_procedure_plan', goal: 'Review', operations: [{ operation: { name: 'query_lap_analysis_result', arguments: {} } }] } },
        { workflow: { name: 'set_procedure_plan', goal: 'Review', operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query', arguments: [] } }] } },
        { ...procedure('query_lap_analysis_result'), requests: [] },
        { workflow: { ...procedure('query_lap_analysis_result').workflow, requests: [] } },
        { workflow: { name: 'set_procedure_plan', goal: 'Review', operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query', arguments: {} }, show_map: { title: 'Map', arguments: {} } }] } },
        ...['payload', 'args', 'parameters', 'name', 'status'].map((alias) => ({
            workflow: { name: 'set_procedure_plan', goal: 'Review', operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query', arguments: {}, [alias]: {} } }] },
        })),
    ])('rejects malformed or legacy procedure input %#', (input) => {
        expect(() => parseProcedurePlanInput(input)).toThrow();
    });

    it('forwards argument keys literally instead of unwrapping former aliases', () => {
        const args = { arguments: { a: 1 }, args: { b: 2 }, parameters: { c: 3 }, tools: ['literal'], operation: { name: 'literal' } };
        expect(parseProcedurePlanInput({ workflow: { name: 'set_procedure_plan',
            goal: 'Review', operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query', arguments: args } }],
        } }).requests[0].payload).toEqual(args);
    });

    it.each(['', '   '])('defaults a present blank procedure goal to the first title (%j)', (goal) => {
        expect(parseProcedurePlanInput({ workflow: { name: 'set_procedure_plan',
            goal, operations: [{ operation: { name: 'query_lap_analysis_result', title: 'First query', arguments: {} } }],
        } }).goal).toBe('First query');
    });

    it('requires own envelope, goal, and procedure arguments properties', () => {
        expect(() => parseRepeatablePlanInput(Object.create(repeatable()))).toThrow();
        expect(() => parseProcedurePlanInput(Object.create(procedure('query_lap_analysis_result')))).toThrow();
        expect(() => parseProcedurePlanInput({ workflow: Object.assign(Object.create({ goal: 'Inherited' }), {
            operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query', arguments: {} } }],
        }) })).toThrow();
        expect(() => parseProcedurePlanInput({ workflow: { name: 'set_procedure_plan',
            goal: 'Review', operations: [{ operation: { name: 'query_lap_analysis_result', ...Object.assign(Object.create({ arguments: {} }), { title: 'Query' }) } }],
        } })).toThrow();
    });

    it.each([
        { name: 'Legacy', steps: [], stop_when: { tool: { name: 'query_lap_analysis_result' }, operator: 'eq', target: 0 } },
        { ...repeatable(), steps: [] },
        { workflow: { ...repeatable().workflow, steps: [] } },
        { workflow: { ...repeatable().workflow, goal: '' } },
        { workflow: { ...repeatable().workflow, operations: [] } },
        { workflow: { ...repeatable().workflow, operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Query' } }] } },
        { workflow: { ...repeatable().workflow, operations: [{ operation: { name: 'query_lap_analysis_result', id: 'query' } }] } },
        { workflow: { ...repeatable().workflow, operations: [{ operation: { name: 'query_lap_analysis_result', id: 'query', title: 'Query', arguments: [] } }] } },
        { workflow: { ...repeatable().workflow, stop_when: { tool: { query_lap_analysis_result: {} }, operator: 'eq', target: 0 } } },
        { workflow: { ...repeatable().workflow, stop_when: { tool: { name: 'query_lap_analysis_result', }, operator: 'eq', target: '0' } } },
    ])('rejects malformed or legacy repeatable input %#', (input) => {
        expect(() => parseRepeatablePlanInput(input)).toThrow();
    });

    it('defaults omitted repeatable step and stop arguments to empty objects at dispatch', async () => {
        const dispatch = Object.assign(jest.fn(() => asTool(createOperation({ status: 'ready', data: 0 }, 'complete'))), { validate: jest.fn() });
        const runner = new RepeatablePlanRunner('repeatable', dispatch);
        await expect(runner.create(repeatable()).result).resolves.toMatchObject({ status: 'achieved' });
        expect(dispatch.mock.calls).toEqual([['query_lap_analysis_result', {}, undefined, runner], ['query_lap_analysis_result', {}, undefined, runner]]);
        expect(dispatch.validate.mock.calls).toEqual([['query_lap_analysis_result'], ['query_lap_analysis_result']]);
        runner.dispose();
    });
});

describe('workflow replacement preflight', () => {
    it('preserves a live running task and inserts no events when a later input is invalid', () => {
        const tool = asTool(createControlledOperation<Record<string, unknown>>().operation);
        const abort = jest.spyOn(tool, 'abort');
        const changed = jest.fn();
        const runner = new LiveRangeTodoListRunner('live', changed);
        const event = (id: string): LiveRangeTodoEventInput => ({
            id, normalized_position: 0.2, lead_time_seconds: 0, content: { title: id }, taskStart: () => tool,
        });
        runner.addEvent(event('active'));
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0 });
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.3 });
        const snapshot = runner.getSnapshot();
        changed.mockClear();
        const invalid = { ...event('invalid'), taskStart: undefined } as unknown as LiveRangeTodoEventInput;
        expect(() => runner.addEvent(invalid)).toThrow();
        expect(() => runner.replaceEvents([event('replacement'), invalid])).toThrow();
        expect(() => runner.updateEvents([{ id: 'active', content: { title: 'Changed' } }, { id: 'missing', content: { title: 'Missing' } }])).toThrow();
        expect(runner.getSnapshot()).toEqual(snapshot);
        expect(changed).not.toHaveBeenCalled();
        expect(abort).not.toHaveBeenCalled();
        runner.dispose();
    });

    it.each(['legacy', 'later tool'])('preserves an active procedure for invalid %s input', async (failure) => {
        const controller = createControlledOperation<Record<string, unknown>>();
        const tool = asTool(controller.operation);
        const abort = jest.spyOn(tool, 'abort');
        const dispatch = Object.assign(jest.fn(() => tool), { validate: jest.fn((name: string) => {
            if (name === 'forbidden') throw new Error('Forbidden tool');
        }) });
        const changed = jest.fn();
        const runner = new ProcedurePlanRunner('procedure', dispatch, changed);
        const active = runner.createProcedurePlan(procedure('query_lap_analysis_result'));
        void active.result.catch(() => undefined);
        const snapshot = runner.getSnapshot();
        changed.mockClear();
        const invalid = failure === 'legacy' ? snapshot : procedure('query_lap_analysis_result', 'forbidden');
        await expect(runner.createProcedurePlan(invalid as ProcedurePlanInput).result).rejects.toThrow();
        expect(abort).not.toHaveBeenCalled();
        expect(dispatch).toHaveBeenCalledTimes(1);
        expect(changed).not.toHaveBeenCalled();
        expect(runner.getSnapshot()).toEqual(snapshot);
        runner.dispose();
    });

    it.each(['legacy', 'later tool', 'stop tool'])('preserves an active repeatable plan for invalid %s input', async (failure) => {
        const tool = asTool(createControlledOperation<Record<string, unknown>>().operation);
        const abort = jest.spyOn(tool, 'abort');
        const dispatch = Object.assign(jest.fn(() => tool), { validate: jest.fn((name: string) => {
            if (name === 'forbidden') throw new Error('Forbidden tool');
        }) });
        const changed = jest.fn();
        const runner = new RepeatablePlanRunner('repeatable', dispatch, changed);
        const active = runner.createRepeatablePlan(repeatable());
        void active.result.catch(() => undefined);
        const snapshot = runner.getSnapshot();
        changed.mockClear();
        const invalid = failure === 'legacy' ? snapshot : failure === 'later tool'
            ? repeatable(['query_lap_analysis_result', 'forbidden']) : repeatable(undefined, 'forbidden');
        await expect(runner.createRepeatablePlan(invalid as RepeatablePlanInput).result).rejects.toThrow();
        expect(abort).not.toHaveBeenCalled();
        expect(dispatch).toHaveBeenCalledTimes(1);
        expect(changed).not.toHaveBeenCalled();
        expect(runner.getSnapshot()).toEqual(snapshot);
        runner.dispose();
    });
});

describe('workflow children at runner boundaries', () => {
    it('accepts workflow steps with a tool stop check without changing their kind', async () => {
        const operation = asWorkflow(createOperation({ status: 'ready', data: 0 }, 'complete'));
        const stop = asTool(createOperation({ status: 'ready', data: 0 }, 'complete'));
        const dispatch = Object.assign(jest.fn()
            .mockReturnValueOnce(operation).mockReturnValue(stop), { validate: jest.fn() });
        const runner = new RepeatablePlanRunner('repeatable', dispatch);
        await expect(runner.create(repeatable()).result).resolves.toMatchObject({ status: 'achieved' });
        expect(operation.kind).toBe('workflow');
        expect(runner.getSnapshot()).toBeNull();
        runner.dispose();
    });

    it('rejects a workflow stop check before executing any steps', async () => {
        const dispatch = Object.assign(jest.fn(), { validate: jest.fn() });
        const runner = new RepeatablePlanRunner('repeatable', dispatch);
        await expect(runner.create(repeatable(undefined, 'set_procedure_plan')).result)
            .rejects.toThrow(/stop condition/);
        expect(dispatch).not.toHaveBeenCalled();
        expect(runner.getSnapshot()).toBeNull();
        runner.dispose();
    });
});
