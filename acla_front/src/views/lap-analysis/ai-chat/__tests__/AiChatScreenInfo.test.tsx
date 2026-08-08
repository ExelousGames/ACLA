import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import AiChatScreenInfo from '../AiChatScreenInfo';

const info = {
    title: 'Live at Monza',
    description: 'Current telemetry and coaching workspace.',
    status: { label: 'Recording', tone: 'success' as const },
    facts: [
        { label: 'Track', value: 'Monza' },
        { label: 'Car', value: 'BMW M4 GT3' },
    ],
};

describe('AiChatScreenInfo', () => {
    it('shows curated screen information on pointer hover and keeps it open over the card', () => {
        const { container } = render(<AiChatScreenInfo label="Live Session" info={info} />);
        const wrapper = container.querySelector('.ai-chat__screen-info') as HTMLElement;
        const trigger = screen.getByRole('button', { name: 'Screen information: Live Session' });

        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        fireEvent.pointerEnter(wrapper);
        expect(screen.getByRole('dialog', { name: 'Live Session information' })).toHaveTextContent('Monza');

        fireEvent.pointerEnter(screen.getByRole('dialog'));
        expect(screen.getByRole('dialog')).toBeInTheDocument();

        fireEvent.pointerLeave(wrapper);
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('supports focus, Escape, blur, and click toggling', () => {
        render(<AiChatScreenInfo label="User Summary" info={info} />);
        const trigger = screen.getByRole('button', { name: 'Screen information: User Summary' });

        fireEvent.focus(trigger);
        expect(trigger).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByRole('dialog')).toHaveTextContent('Current telemetry and coaching workspace.');

        fireEvent.keyDown(trigger, { key: 'Escape' });
        expect(trigger).toHaveAttribute('aria-expanded', 'false');

        fireEvent.click(trigger);
        expect(trigger).toHaveAttribute('aria-expanded', 'true');
        fireEvent.click(trigger);
        expect(trigger).toHaveAttribute('aria-expanded', 'false');

        fireEvent.focus(trigger);
        fireEvent.blur(trigger, { relatedTarget: document.body });
        expect(trigger).toHaveAttribute('aria-expanded', 'false');
    });

    it('opens on the first click for touch-style interaction', () => {
        render(<AiChatScreenInfo label="Recorded Session" info={info} />);
        const trigger = screen.getByRole('button', { name: 'Screen information: Recorded Session' });

        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        fireEvent.click(trigger);
        expect(screen.getByRole('dialog')).toBeInTheDocument();
    });
});
