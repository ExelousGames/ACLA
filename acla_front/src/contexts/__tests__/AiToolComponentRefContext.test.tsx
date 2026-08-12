import React, { StrictMode, useMemo } from 'react';
import { act, render } from '@testing-library/react';
import {
    ComponentMountTimeoutError,
    ComponentNameMismatchError,
    DuplicateComponentNameError,
    AiToolComponentRefProvider,
    awaitNamedComponentHandle,
    createAiToolComponentRefDirectory,
    useAiToolComponentRefDirectory,
    useRegisterAiToolComponentRef,
} from '../AiToolComponentRefContext';

const handle = (name: string) => ({ getComponentName: () => name });

describe('AiToolComponentRefDirectory', () => {
    it('registers, finds, updates for the same owner, and releases only that owner', () => {
        const onChange = jest.fn();
        const directory = createAiToolComponentRefDirectory(onChange);
        const owner = Symbol('owner');
        const other = Symbol('other');
        const first = handle('component');
        const second = handle('component');

        const ref = directory.reserveComponentRef('component', owner, first);
        expect(directory.findComponentRef('component')).toBe(ref);
        expect(ref.current).toBe(first);
        expect(directory.reserveComponentRef('component', owner, second)).toBe(ref);
        expect(ref.current).toBe(second);
        expect(onChange).toHaveBeenCalledTimes(1);
        expect(directory.releaseComponentRef('component', other)).toBe(false);
        expect(ref.current).toBe(second);
        expect(directory.releaseComponentRef('component', owner)).toBe(true);
        expect(onChange).toHaveBeenCalledTimes(2);
        expect(ref.current).toBeNull();
        expect(directory.findComponentRef('component')).toBeNull();
    });

    it('rejects duplicate owners and reported-name mismatches', () => {
        const directory = createAiToolComponentRefDirectory();
        directory.reserveComponentRef('component', Symbol('first'), handle('component'));

        expect(() => directory.reserveComponentRef('component', Symbol('second'), handle('component')))
            .toThrow(DuplicateComponentNameError);
        expect(() => directory.reserveComponentRef('claimed', Symbol('owner'), handle('reported')))
            .toThrow(ComponentNameMismatchError);
    });

    it('awaits registration without polling and times out with the named error', async () => {
        jest.useFakeTimers();
        const directory = createAiToolComponentRefDirectory();
        const awaiting = directory.awaitComponentRef('child');
        const child = handle('child');
        directory.reserveComponentRef('child', Symbol('owner'), child);
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
    useRegisterAiToolComponentRef(name, registeredHandle);
    return null;
};

const RegisteredWithFreshHandle = ({ name }: { name: string }) => {
    useRegisterAiToolComponentRef(name, { getComponentName: () => name });
    return null;
};

const Observer = ({ onDirectory }: { onDirectory: (directory: ReturnType<typeof useAiToolComponentRefDirectory>) => void }) => {
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

    it('does not enter an update loop when a component refreshes its handle on render', () => {
        expect(() => render(
            <AiToolComponentRefProvider>
                <RegisteredWithFreshHandle name="fresh" />
            </AiToolComponentRefProvider>,
        )).not.toThrow();
    });

    it('supports Strict Mode replay by one owner and publishes fresh handles', () => {
        let directory: ReturnType<typeof useAiToolComponentRefDirectory> | null = null;
        const view = render(
            <StrictMode>
                <AiToolComponentRefProvider>
                    <Registered name="stable" value={1} />
                    <Observer onDirectory={(value) => { directory = value; }} />
                </AiToolComponentRefProvider>
            </StrictMode>,
        );
        expect((directory!.findComponentRef('stable')!.current as any).getValue()).toBe(1);

        view.rerender(
            <StrictMode>
                <AiToolComponentRefProvider>
                    <Registered name="stable" value={7} />
                    <Observer onDirectory={(value) => { directory = value; }} />
                </AiToolComponentRefProvider>
            </StrictMode>,
        );
        expect((directory!.findComponentRef('stable')!.current as any).getValue()).toBe(7);
    });

    it('unregisters the previous workflow when a keyed child is replaced', () => {
        let directory: ReturnType<typeof useAiToolComponentRefDirectory> | null = null;
        const view = render(
            <AiToolComponentRefProvider>
                <Registered key="goal" name="goal" value={1} />
                <Observer onDirectory={(value) => { directory = value; }} />
            </AiToolComponentRefProvider>,
        );
        expect(directory!.getComponentNames()).toContain('goal');

        view.rerender(
            <AiToolComponentRefProvider>
                <Registered key="procedure-plan" name="procedure-plan" value={2} />
                <Observer onDirectory={(value) => { directory = value; }} />
            </AiToolComponentRefProvider>,
        );

        expect(directory!.findComponentRef('goal')).toBeNull();
        expect(directory!.getComponentNames()).toContain('procedure-plan');
    });

    it('requires a keyed remount when a mounted name changes', () => {
        const view = render(
            <AiToolComponentRefProvider>
                <Registered name="first" value={1} />
            </AiToolComponentRefProvider>,
        );
        expect(() => view.rerender(
            <AiToolComponentRefProvider>
                <Registered name="second" value={1} />
            </AiToolComponentRefProvider>,
        )).toThrow(ComponentNameMismatchError);
    });
});
