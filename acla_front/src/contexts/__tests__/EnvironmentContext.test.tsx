import React from 'react';
import { render, screen } from '@testing-library/react';

jest.mock('utils/environment', () => ({
    detectEnvironment: jest.fn().mockReturnValue('web'),
}));

import EnvironmentProvider, { useEnvironment } from 'contexts/EnvironmentContext';
import { detectEnvironment } from 'utils/environment';

const mockedDetect = detectEnvironment as jest.Mock;

const TestConsumer = () => {
    const env = useEnvironment();
    return <div data-testid="env">{env}</div>;
};

describe('EnvironmentContext', () => {
    it('should provide the detected environment to children', () => {
        mockedDetect.mockReturnValue('web');

        render(
            <EnvironmentProvider>
                <TestConsumer />
            </EnvironmentProvider>
        );

        expect(screen.getByTestId('env')).toHaveTextContent('web');
    });

    it('should provide "electron" when detected', () => {
        mockedDetect.mockReturnValue('electron');

        render(
            <EnvironmentProvider>
                <TestConsumer />
            </EnvironmentProvider>
        );

        expect(screen.getByTestId('env')).toHaveTextContent('electron');
    });
});
