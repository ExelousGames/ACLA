import { captureGameScreen } from './screen-capture';

const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');
let getDisplayMedia: jest.Mock;
let selectSource: jest.Mock;

beforeEach(() => {
    getDisplayMedia = jest.fn().mockResolvedValue({});
    selectSource = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getDisplayMedia } });
    window.screenCapture = { listSources: jest.fn(), selectSource };
});

afterEach(() => {
    delete window.screenCapture;
    if (originalMediaDevices) Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices);
    else Reflect.deleteProperty(navigator, 'mediaDevices');
});

it('rejects capture without Electron even when display capture is available', async () => {
    delete window.screenCapture;
    await expect(captureGameScreen('window:42')).rejects.toThrow('only in the Electron desktop app');
    expect(getDisplayMedia).not.toHaveBeenCalled();
});

it('requires a selected desktop source before requesting capture', async () => {
    await expect(captureGameScreen('')).rejects.toThrow('Choose a game window or screen first.');
    expect(selectSource).not.toHaveBeenCalled();
    expect(getDisplayMedia).not.toHaveBeenCalled();
});

it('waits for Electron to grant the selected source before capturing without audio', async () => {
    let grantSource!: () => void;
    selectSource.mockReturnValue(new Promise<void>((resolve) => { grantSource = resolve; }));
    const capture = captureGameScreen('window:42');
    expect(selectSource).toHaveBeenCalledWith('window:42');
    expect(getDisplayMedia).not.toHaveBeenCalled();
    grantSource();
    await capture;
    expect(getDisplayMedia).toHaveBeenCalledTimes(1);
    expect(getDisplayMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
});

it('does not request display capture when Electron rejects the selected source', async () => {
    selectSource.mockRejectedValue(new Error('The selected window is no longer available.'));
    await expect(captureGameScreen('window:42')).rejects.toThrow('selected window is no longer available');
    expect(getDisplayMedia).not.toHaveBeenCalled();
});
