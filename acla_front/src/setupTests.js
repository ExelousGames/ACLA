// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Polyfill TextEncoder/TextDecoder for react-router v7 in JSDOM
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

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
