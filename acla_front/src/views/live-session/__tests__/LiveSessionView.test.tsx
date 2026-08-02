import React from 'react';
import { render, screen } from '@testing-library/react';

jest.mock('../LiveTelemetryWorkspace', () => () => <div>Live workspace</div>);

import LiveSessionView from '../LiveSessionView';

describe('LiveSessionView', () => {
    it('contains the recorder host within the Live Session content area', () => {
        render(<LiveSessionView />);

        const view = screen.getByRole('region', { name: 'Live Session' });
        expect(view).toContainElement(screen.getByTestId('live-session-recorder-host'));
        expect(view).toHaveTextContent('Live workspace');
    });
});
