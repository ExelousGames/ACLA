import React from 'react';
import { act, render } from '@testing-library/react';
import { Tts, type TtsHandle } from './Tts';
import { audioManager } from 'services/audio';
import { installAudioDoubles } from 'services/audio/test-audio';

const pack = Object.freeze({ text: 'Saved speech', audioDataUrl: 'data:audio/wav;base64,c2F2ZWQ=', durationMs: 2000 });

it('retains the imperative API, reports discard, and replays the caller-owned pack explicitly', async () => {
    const audio = installAudioDoubles();
    const ref = React.createRef<TtsHandle>();
    const onEnded = jest.fn();
    const onComplete = jest.fn();
    const view = render(<Tts ref={ref} pack={pack} priority={60} volume={0.7} onEnded={onEnded} onComplete={onComplete} />);
    expect(audio.play).not.toHaveBeenCalled();
    await act(async () => { ref.current!.play(); });
    expect(audio.media[0].volume).toBe(0.7);
    const lower = audioManager.play({ type: 'voice', priority: 50, url: 'other.wav' });
    expect(lower.outcome?.status).toBe('rejected');
    const replacement = audioManager.play({ type: 'voice', priority: 60, url: 'other.wav' });
    await act(async () => { await Promise.resolve(); });
    expect(onComplete).toHaveBeenCalledWith({ status: 'replaced' });
    expect(onEnded).not.toHaveBeenCalled();
    replacement.stop();
    await act(async () => { ref.current!.play(); });
    expect(audio.media[2].src).toBe(pack.audioDataUrl);
    await act(async () => { audio.media[2].dispatchEvent(new Event('ended')); });
    expect(onEnded).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenLastCalledWith({ status: 'completed' });
    view.unmount();
    audio.restore();
});

it('isolates replacement, stop and unmount from stale play promise failures', async () => {
    const audio = installAudioDoubles();
    const failures: Array<(error: Error) => void> = [];
    audio.play.mockImplementation(() => new Promise<void>((_, reject) => { failures.push(reject); }));
    const ref = React.createRef<TtsHandle>();
    const onError = jest.fn();
    const onComplete = jest.fn();
    const view = render(<Tts ref={ref} pack={pack} onError={onError} onComplete={onComplete} />);
    act(() => { ref.current!.play(); ref.current!.play(); ref.current!.stop(); ref.current!.play(); });
    view.rerender(<Tts ref={ref} pack={{ ...pack, audioDataUrl: `${pack.audioDataUrl}AA==` }} onError={onError} />);
    act(() => { ref.current!.play(); });
    view.unmount();
    await act(async () => { failures.forEach((reject) => reject(new Error('stale'))); });
    expect(onError).not.toHaveBeenCalled();
    expect(onComplete).not.toHaveBeenCalled();
    expect(audio.pause).toHaveBeenCalledTimes(4);
    expect(audio.media.every((element) => !element.hasAttribute('src'))).toBe(true);
    audio.restore();
});
