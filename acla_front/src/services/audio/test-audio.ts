interface TestAudioNode {
    buffer: AudioBuffer | null;
    onended: (() => void) | null;
    connect: jest.Mock; disconnect: jest.Mock; start: jest.Mock; stop: jest.Mock;
}

/** Browser audio doubles shared by lifecycle tests. */
export class TestAudioContext {
    static instances: TestAudioContext[] = [];
    static initialState: AudioContextState = 'running';
    state = TestAudioContext.initialState;
    currentTime = 0;
    destination = {};
    onstatechange: (() => void) | null = null;
    nodes: TestAudioNode[] = [];
    gain = { gain: { value: 1 }, connect: jest.fn(), disconnect: jest.fn() };
    resolveResume!: () => void;
    rejectResume!: (error: Error) => void;
    resume = jest.fn(() => new Promise<void>((resolve, reject) => {
        this.resolveResume = () => { this.state = 'running'; resolve(); };
        this.rejectResume = reject;
    }));
    close = jest.fn(async () => { this.state = 'closed'; });
    createGain = jest.fn(() => this.gain);
    createBuffer = jest.fn((_channels: number, length: number, sampleRate: number) => ({
        duration: length / sampleRate, copyToChannel: jest.fn(),
    }));
    createBufferSource = jest.fn(() => {
        const node = {
            buffer: null as AudioBuffer | null, onended: null as (() => void) | null,
            connect: jest.fn(), disconnect: jest.fn(), start: jest.fn(), stop: jest.fn(),
        };
        this.nodes.push(node);
        return node;
    });
    constructor(public options?: AudioContextOptions) { TestAudioContext.instances.push(this); }
}

export const installAudioDoubles = () => {
    const media: HTMLAudioElement[] = [];
    const play = jest.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
    const pause = jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
    const load = jest.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => undefined);
    jest.spyOn(globalThis, 'Audio').mockImplementation((url?: string) => {
        const element = document.createElement('audio');
        if (url) element.src = url;
        media.push(element);
        return element;
    });
    const contextDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'AudioContext');
    Object.defineProperty(globalThis, 'AudioContext', { configurable: true, value: TestAudioContext });
    TestAudioContext.instances = [];
    TestAudioContext.initialState = 'running';
    const createUrlDescriptor = Object.getOwnPropertyDescriptor(URL, 'createObjectURL');
    const revokeUrlDescriptor = Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL');
    let sequence = 0;
    const createObjectURL = jest.fn(() => `blob:test-${++sequence}`);
    const revokeObjectURL = jest.fn();
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL });
    const restoreProperty = (object: object, key: string, descriptor?: PropertyDescriptor) => {
        if (descriptor) Object.defineProperty(object, key, descriptor);
        else Reflect.deleteProperty(object, key);
    };
    return {
        media, play, pause, load, createObjectURL, revokeObjectURL,
        restore: () => {
            jest.restoreAllMocks();
            restoreProperty(globalThis, 'AudioContext', contextDescriptor);
            restoreProperty(URL, 'createObjectURL', createUrlDescriptor);
            restoreProperty(URL, 'revokeObjectURL', revokeUrlDescriptor);
        },
    };
};
