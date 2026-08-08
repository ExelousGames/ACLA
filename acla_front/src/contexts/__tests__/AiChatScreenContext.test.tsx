import React, { useMemo, useRef } from 'react';
import { act, render, screen } from '@testing-library/react';
import {
    AiChatScreenHandle,
    AiChatScreenProvider,
    useAiChatScreen,
    useAiChatScreenRegistration,
} from '../AiChatScreenContext';

const observedRefs: Array<React.MutableRefObject<AiChatScreenHandle | null>> = [];

const RegisteredScreen = ({ label, value }: { label: string; value: number }) => {
    const valueRef = useRef(value);
    valueRef.current = value;
    const componentRef = useRef<AiChatScreenHandle | null>(null);
    if (componentRef.current === null) {
        componentRef.current = {
            getAiContext: () => ({ value: valueRef.current }),
            getToolHandlers: () => ({}),
        };
        observedRefs.push(componentRef);
    }
    const registration = useMemo(() => ({
        screenId: label.toLowerCase(),
        assistantMode: 'front_desk' as const,
        pillLabel: label,
        componentRef,
        getPillInfo: () => ({
            title: label,
            description: `${label} description`,
            status: { label: 'Ready', tone: 'success' as const },
            facts: [{ label: 'Value', value: String(value) }],
        }),
    }), [label, value]);
    useAiChatScreenRegistration(registration);
    return null;
};

const Observer = () => {
    const { activeScreen } = useAiChatScreen();
    return (
        <div>
            <span data-testid="active-label">{activeScreen?.pillLabel || 'none'}</span>
            <span data-testid="active-value">
                {String(activeScreen?.componentRef.current?.getAiContext().value ?? 'none')}
            </span>
        </div>
    );
};

const Harness = ({ showFirst, showSecond, value }: {
    showFirst: boolean;
    showSecond: boolean;
    value: number;
}) => (
    <AiChatScreenProvider>
        {showFirst && <RegisteredScreen label="First" value={value} />}
        {showSecond && <RegisteredScreen label="Second" value={2} />}
        <Observer />
    </AiChatScreenProvider>
);

describe('AiChatScreenProvider', () => {
    beforeEach(() => {
        observedRefs.length = 0;
    });

    it('replaces the active registration and guards cleanup by owner', () => {
        const view = render(<Harness showFirst showSecond value={1} />);
        expect(screen.getByTestId('active-label')).toHaveTextContent('Second');

        view.rerender(<Harness showFirst={false} showSecond value={1} />);
        expect(screen.getByTestId('active-label')).toHaveTextContent('Second');

        view.rerender(<Harness showFirst={false} showSecond={false} value={1} />);
        expect(screen.getByTestId('active-label')).toHaveTextContent('none');
    });

    it('publishes current values while keeping a stable component ref', () => {
        const view = render(<Harness showFirst showSecond={false} value={1} />);
        const firstRef = observedRefs[0];
        expect(screen.getByTestId('active-value')).toHaveTextContent('1');

        act(() => {
            view.rerender(<Harness showFirst showSecond={false} value={7} />);
        });

        expect(screen.getByTestId('active-value')).toHaveTextContent('7');
        expect(observedRefs[0]).toBe(firstRef);
        expect(observedRefs).toHaveLength(1);
    });
});
