import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import CaptureBoundaryStart from './CaptureBoundaryStart';

it('places a horizontal scan line in the capture without calibration or depth', () => {
    const onChange = jest.fn();
    const { rerender } = render(<CaptureBoundaryStart width={1600} height={900} value={0.6} onChange={onChange} />);
    const line = screen.getByRole('slider', { name: 'Boundary start line' });
    expect(line.querySelector('path')).toHaveAttribute('d', 'M0,270 L800,270');
    expect(line).toHaveAttribute('aria-valuenow', '60');
    expect(screen.getByRole('slider', { name: 'Boundary start' })).toHaveValue('60');
    expect(screen.queryByLabelText('Left boundary start')).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('slider', { name: 'Boundary start' }), { target: { value: '75' } });
    expect(onChange).toHaveBeenLastCalledWith(0.75);
    rerender(<CaptureBoundaryStart width={900} height={1600} value={0.6} onChange={onChange} />);
    const viewHeight = 800 * 1600 / 900;
    expect(line.querySelector('path')).toHaveAttribute('d', `M0,${viewHeight * 0.6} L800,${viewHeight * 0.6}`);
});
