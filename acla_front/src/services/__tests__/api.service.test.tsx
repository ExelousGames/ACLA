import axios from 'axios';

// Use a global container to capture interceptor callbacks from the hoisted jest.mock
const _capture: { requestInterceptor?: Function } = {};
(global as any).__axisMockCapture = _capture;

jest.mock('axios', () => {
    const capture = (global as any).__axisMockCapture;
    const instance = {
        get: jest.fn(),
        post: jest.fn(),
        put: jest.fn(),
        delete: jest.fn(),
        interceptors: {
            request: {
                use: jest.fn((onFulfilled: Function) => {
                    capture.requestInterceptor = onFulfilled;
                }),
            },
            response: { use: jest.fn() },
        },
    };
    return {
        __esModule: true,
        default: {
            create: jest.fn(() => instance),
            __mockInstance: instance,
        },
    };
});

import { ApiService } from 'services/api.service';

const mockedAxios = axios as any;
const mockAxiosInstance = mockedAxios.__mockInstance;

describe('ApiService', () => {
    // Create once — CRA's resetMocks: true clears mock implementations between tests
    const apiService = new ApiService();

    beforeEach(() => {
        localStorage.clear();
        mockAxiosInstance.get.mockReset();
        mockAxiosInstance.post.mockReset();
        mockAxiosInstance.put.mockReset();
        mockAxiosInstance.delete.mockReset();
    });

    it('should create an axios instance', () => {
        // ApiService calls axios.create in its constructor
        expect(apiService).toBeDefined();
    });

    describe('get', () => {
        it('should call axiosInstance.get with correct params', async () => {
            const mockResponse = { data: { id: 1 }, status: 200 };
            mockAxiosInstance.get.mockResolvedValue(mockResponse);

            const result = await apiService.get('/test', { page: 1 });

            expect(mockAxiosInstance.get).toHaveBeenCalledWith('/test', { params: { page: 1 } });
            expect(result).toEqual(mockResponse);
        });

        it('should throw on error', async () => {
            const error = { message: 'Network error', status: 500 };
            mockAxiosInstance.get.mockRejectedValue(error);

            await expect(apiService.get('/test')).rejects.toEqual(error);
        });
    });

    describe('post', () => {
        it('should call axiosInstance.post with correct data', async () => {
            const mockResponse = { data: { success: true }, status: 201 };
            mockAxiosInstance.post.mockResolvedValue(mockResponse);

            const result = await apiService.post('/test', { name: 'test' });

            expect(mockAxiosInstance.post).toHaveBeenCalledWith('/test', { name: 'test' });
            expect(result).toEqual(mockResponse);
        });
    });

    describe('put', () => {
        it('should call axiosInstance.put with correct data', async () => {
            const mockResponse = { data: { updated: true }, status: 200 };
            mockAxiosInstance.put.mockResolvedValue(mockResponse);

            const result = await apiService.put('/test/1', { name: 'updated' });

            expect(mockAxiosInstance.put).toHaveBeenCalledWith('/test/1', { name: 'updated' });
            expect(result).toEqual(mockResponse);
        });
    });

    describe('delete', () => {
        it('should call axiosInstance.delete', async () => {
            const mockResponse = { data: null, status: 204 };
            mockAxiosInstance.delete.mockResolvedValue(mockResponse);

            const result = await apiService.delete('/test/1');

            expect(mockAxiosInstance.delete).toHaveBeenCalledWith('/test/1');
            expect(result).toEqual(mockResponse);
        });
    });

    describe('request interceptor - auth token', () => {
        it('should attach Bearer token from localStorage', () => {
            localStorage.setItem('token', 'my-jwt-token');

            const config = { headers: {} as any };
            const result = _capture.requestInterceptor!(config);

            expect(result.headers.Authorization).toBe('Bearer my-jwt-token');
        });

        it('should not attach token when none exists', () => {
            const config = { headers: {} as any };
            const result = _capture.requestInterceptor!(config);

            expect(result.headers.Authorization).toBeUndefined();
        });
    });
});
