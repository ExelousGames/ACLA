import React from 'react';
import { render, screen } from '@testing-library/react';
import BaselineProgressDisplay, { baselineProgressDisplayOverlayRenderer } from '../BaselineProgressDisplay';
import type { BaselineCollectionTag } from '../BaselineCollection';

const recordingTag: BaselineCollectionTag = {
    status: 'collecting',
    progress_percent: 42,
    detail: 'Recording the current lap.',
    track: 'brands_hatch',
    car: 'Ferrari 296',
    current_lap: 5,
    baseline_lap_id: 5,
};

describe('BaselineProgressDisplay', () => {
    it('omits the track and car from the main panel', () => {
        render(<BaselineProgressDisplay tag={recordingTag} />);

        expect(screen.getByText('Baseline run')).toBeInTheDocument();
        expect(screen.getByText('Recording stage')).toBeInTheDocument();
        expect(screen.queryByText('Brands Hatch')).not.toBeInTheDocument();
        expect(screen.queryByText('Ferrari 296')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Baseline session details')).not.toBeInTheDocument();
    });

    it('omits redundant labels and session metadata from the overlay', () => {
        render(baselineProgressDisplayOverlayRenderer.renderOverlay(
            recordingTag,
            'expanded',
            {} as any,
        ));

        expect(screen.queryByText('Baseline run')).not.toBeInTheDocument();
        expect(screen.queryByText('Recording stage')).not.toBeInTheDocument();
        expect(screen.queryByText('Brands Hatch')).not.toBeInTheDocument();
        expect(screen.queryByText('Ferrari 296')).not.toBeInTheDocument();
        expect(screen.getByText('Recording baseline')).toBeInTheDocument();
        expect(screen.getByText('Recording the current lap.')).toBeInTheDocument();
    });
});
