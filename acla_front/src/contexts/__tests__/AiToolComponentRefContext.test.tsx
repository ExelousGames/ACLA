import React, { StrictMode, useMemo, useRef } from 'react';
import { act, render } from '@testing-library/react';
import {
    ComponentMountTimeoutError,
    DuplicateComponentNameError,
    AiToolComponentRefProvider,
    awaitNamedComponentHandle,
    createAiToolComponentRefDirectory,
    useAiToolComponentRefDirectory,
    useRegisterAiToolComponentRef,
} from '../AiToolComponentRefContext';

const handle = (name: string) => ({ getComponentName: () => name });

describe('AiToolComponentRefDirectory', () => {
    it('registers and unregisters the exact creator reference', () => {
        const onChange = jest.fn();
        const directory = createAiToolComponentRefDirectory(onChange);
        const first = handle('component');
        const ref = { current: first };

        directory.registerComponentRef(ref);
        expect(directory.findComponentRef('component')).toBe(ref);
        expect(directory.getComponentRefs()).toEqual([ref]);

        const updated = handle('component');
        ref.current = updated;
        directory.registerComponentRef(ref);
        expect(directory.findComponentRef('component')?.current).toBe(updated);
        expect(onChange).toHaveBeenCalledTimes(1);

        expect(directory.unregisterComponentRef({ current: updated })).toBe(false);
        expect(directory.unregisterComponentRef(ref)).toBe(true);
        expect(directory.findComponentRef('component')).toBeNull();
        expect(onChange).toHaveBeenCalledTimes(2);
    });

    it('derives identity only from getComponentName and rejects duplicate live names', () => {
        const directory = createAiToolComponentRefDirectory();
        const first = { current: handle('shared-name') };
        const second = { current: handle('shared-name') };
        directory.registerComponentRef(first);

        expect(() => directory.registerComponentRef(second)).toThrow(DuplicateComponentNameError);
        expect(directory.findComponentRef('shared-name')).toBe(first);
    });

    it('allows many references of one component class when runtime names differ', () => {
        const directory = createAiToolComponentRefDirectory();
        const create = (name: string) => ({ current: { ...handle(name), componentType: 'tool_status' } });
        const first = create('tool-status:run-1');
        const second = create('tool-status:run-2');

        directory.registerComponentRef(first);
        directory.registerComponentRef(second);

        expect(directory.getComponentNames()).toEqual(['tool-status:run-1', 'tool-status:run-2']);
    });

    it('awaits registration without polling and times out with the named error', async () => {
        jest.useFakeTimers();
        const directory = createAiToolComponentRefDirectory();
        const awaiting = directory.awaitComponentRef('child');
        const child = handle('child');
        directory.registerComponentRef({ current: child });
        await expect(awaiting).resolves.toEqual(expect.objectContaining({ current: child }));

        const timeout = directory.awaitComponentRef('missing');
        act(() => jest.advanceTimersByTime(5000));
        await expect(timeout).rejects.toBeInstanceOf(ComponentMountTimeoutError);
        await expect(timeout).rejects.toMatchObject({ componentName: 'missing' });
        jest.useRealTimers();
    });
});

const Registered = ({ name, value }: { name: string; value: number }) => {
    const registeredHandle = useMemo(() => ({
        getComponentName: () => name,
        getValue: () => value,
    }), [name, value]);
    const ref = useRef<typeof registeredHandle | null>(registeredHandle);
    ref.current = registeredHandle;
    useRegisterAiToolComponentRef(ref);
    return null;
};

const RegisteredWithFreshHandle = ({ name }: { name: string }) => {
    const ref = useRef<{ getComponentName(): string } | null>(null);
    ref.current = { getComponentName: () => name };
    useRegisterAiToolComponentRef(ref);
    return null;
};

const Observer = ({ onDirectory }: {
    onDirectory: (directory: ReturnType<typeof useAiToolComponentRefDirectory>) => void;
}) => {
    onDirectory(useAiToolComponentRefDirectory());
    return null;
};

describe('AiToolComponentRefProvider', () => {
    it('returns the current handle when awaiting a first Strict Mode mount replay', async () => {
        let directory: ReturnType<typeof useAiToolComponentRefDirectory> | null = null;
        const view = render(
            <StrictMode>
                <AiToolComponentRefProvider>
                    <Observer onDirectory={(value) => { directory = value; }} />
                </AiToolComponentRefProvider>
            </StrictMode>,
        );
        const awaiting = awaitNamedComponentHandle<any>(directory!, 'strict-mount');

        view.rerender(
            <StrictMode>
                <AiToolComponentRefProvider>
                    <Registered name="strict-mount" value={7} />
                    <Observer onDirectory={(value) => { directory = value; }} />
                </AiToolComponentRefProvider>
            </StrictMode>,
        );

        const currentHandle = directory!.findComponentRef('strict-mount')!.current;
        await expect(awaiting).resolves.toBe(currentHandle);
        expect((currentHandle as any).getValue()).toBe(7);
    });

    it('keeps one stable registration while a component refreshes its handle', () => {
        expect(() => render(
            <AiToolComponentRefProvider>
                <RegisteredWithFreshHandle name="fresh" />
            </AiToolComponentRefProvider>,
        )).not.toThrow();
    });

    it('publishes a fresh current handle through the same stable reference', () => {
        let directory: ReturnType<typeof useAiToolComponentRefDirectory> | null = null;
        const view = render(
            <AiToolComponentRefProvider>
                <Registered name="stable" value={1} />
                <Observer onDirectory={(value) => { directory = value; }} />
            </AiToolComponentRefProvider>,
        );
        const ref = directory!.findComponentRef('stable');
        expect((ref!.current as any).getValue()).toBe(1);

        view.rerender(
            <AiToolComponentRefProvider>
                <Registered name="stable" value={7} />
                <Observer onDirectory={(value) => { directory = value; }} />
            </AiToolComponentRefProvider>,
        );
        expect(directory!.findComponentRef('stable')).toBe(ref);
        expect((ref!.current as any).getValue()).toBe(7);
    });

    it('unregisters a keyed child by its creator reference', () => {
        let directory: ReturnType<typeof useAiToolComponentRefDirectory> | null = null;
        const view = render(
            <AiToolComponentRefProvider>
                <Registered key="goal" name="goal" value={1} />
                <Observer onDirectory={(value) => { directory = value; }} />
            </AiToolComponentRefProvider>,
        );

        view.rerender(
            <AiToolComponentRefProvider>
                <Registered key="procedure-plan" name="procedure-plan" value={2} />
                <Observer onDirectory={(value) => { directory = value; }} />
            </AiToolComponentRefProvider>,
        );

        expect(directory!.findComponentRef('goal')).toBeNull();
        expect(directory!.getComponentNames()).toContain('procedure-plan');
    });
});
