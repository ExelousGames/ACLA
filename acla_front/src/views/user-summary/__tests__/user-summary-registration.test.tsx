import React from 'react';
import { render } from '@testing-library/react';
import {
    OPERATION_COMPONENT_NAMES,
    OperationComponentRefDirectory,
    OperationComponentRefProvider,
    useOperationComponentRefDirectory,
} from 'contexts/OperationComponentRefContext';
import type { UserSummaryHandle } from '../user-summary';

const mockUseUserSummary = jest.fn();
const mockUseAiLabels = jest.fn();

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Element = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    return { Box: Element, Card: Element, Flex: Element, Heading: Element, Separator: () => <hr />, Text: Element, TextArea: (props: any) => <textarea {...props} /> };
});
jest.mock('contexts/UserSummaryContext', () => ({ useUserSummary: () => mockUseUserSummary() }));
jest.mock('contexts/AiLabelsContext', () => ({ useAiLabels: () => mockUseAiLabels() }));
jest.mock('../AnalyzeAllSessionsControl', () => () => <button type="button">Analyze</button>);

import UserSummary from '../user-summary';

let directory: OperationComponentRefDirectory | null = null;
const DirectoryObserver = () => {
    directory = useOperationComponentRefDirectory();
    return null;
};

const summary = { sessionAnalysis: { practice: { tracks: { monza: { trackName: 'Monza', analyzedSessionCount: 3, sections: {} } } } } };
const Harness = () => (
    <OperationComponentRefProvider>
        <UserSummary name={OPERATION_COMPONENT_NAMES.USER_SUMMARY} />
        <DirectoryObserver />
    </OperationComponentRefProvider>
);

describe('UserSummary named component handle', () => {
    beforeEach(() => {
        directory = null;
        mockUseUserSummary.mockReturnValue({ userSummary: summary, userSummaryLoading: false, userSummaryError: '', loadUserSummary: jest.fn() });
        mockUseAiLabels.mockReturnValue({ getLabelName: (id: string) => id, getCategoryLabels: () => [], loading: false, error: '' });
    });

    it('exposes exact identity and fresh summary operations', async () => {
        const view = render(<Harness />);
        const ref = directory!.findComponentRef<UserSummaryHandle>(OPERATION_COMPONENT_NAMES.USER_SUMMARY)!;
        expect(ref.current!.getComponentName()).toBe(OPERATION_COMPONENT_NAMES.USER_SUMMARY);
        await expect(ref.current!.getAvailableUserSummaryMaps().result)
            .resolves.toMatchObject({ status: 'ready', map_count: 1 });

        mockUseUserSummary.mockReturnValue({ userSummary: summary, userSummaryLoading: true, userSummaryError: '', loadUserSummary: jest.fn() });
        view.rerender(<Harness />);
        await expect(ref.current!.getAvailableUserSummaryMaps().result)
            .resolves.toMatchObject({ status: 'loading' });
    });
});
