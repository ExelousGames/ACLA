// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Polyfill TextEncoder/TextDecoder for react-router v7 in JSDOM
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// CodeMirror measures text ranges in the browser. JSDOM does not implement
// these geometry methods, so provide inert measurements for editor tests.
if (typeof Range !== 'undefined' && !Range.prototype.getClientRects) {
    Range.prototype.getClientRects = () => [];
}
if (typeof Range !== 'undefined' && !Range.prototype.getBoundingClientRect) {
    Range.prototype.getBoundingClientRect = () => ({
        bottom: 0,
        height: 0,
        left: 0,
        right: 0,
        top: 0,
        width: 0,
        x: 0,
        y: 0,
        toJSON: () => ({}),
    });
}

// ECharts 6 publishes tree-shaken entrypoints as ESM, while react-scripts 5's
// Jest runtime does not transform that dependency. Component tests provide
// focused mocks where assertions are needed; this safe default keeps modules
// that only import the visualization registry from evaluating a canvas chart.
jest.mock('echarts/charts', () => ({ BarChart: {}, LineChart: {} }));
jest.mock('echarts/components', () => ({
    AriaComponent: {},
    DatasetComponent: {},
    GridComponent: {},
    TooltipComponent: {},
}));
jest.mock('echarts/renderers', () => ({ CanvasRenderer: {} }));
jest.mock('echarts/core', () => ({
    use: jest.fn(),
    init: jest.fn(() => ({
        setOption: jest.fn(),
        resize: jest.fn(),
        dispose: jest.fn(),
    })),
}));
