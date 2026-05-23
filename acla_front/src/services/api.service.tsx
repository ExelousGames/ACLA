import axios, { AxiosError, AxiosInstance, AxiosResponse } from 'axios';


type ApiResponse<T> = {
    data: T;
    status: number;
};

type ApiError = {
    message: string;
    status?: number;
    data?: any;
};

// API methods
export class ApiService {
    private axiosInstance: AxiosInstance;

    constructor() {
        this.axiosInstance = axios.create({
            baseURL: 'http://' + process.env.REACT_APP_BACKEND_SERVER_IP + ":" + process.env.REACT_APP_BACKEND_PROXY_PORT,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        this.setupInterceptors();
    }

    //interceptors are used to modify requests or responses before they are handled by component who called the API
    private setupInterceptors() {
        // Request interceptor
        this.axiosInstance.interceptors.request.use(
            (config) => {
                // Add auth token if exists, token is used for verifying user identity in subsequent requests
                const token = localStorage.getItem('token');
                if (token) {
                    //Authorization: <type> <credentials> pattern --  OAuth 2.0 Authorization Framework
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => Promise.reject(error)
        );

        // Response interceptor
        this.axiosInstance.interceptors.response.use(
            (response) => response,
            (error: AxiosError) => {
                const apiError: ApiError = {
                    message: error.message,
                    status: error.response?.status,
                    data: error.response?.data,
                };
                return Promise.reject(apiError);
            }
        );
    }


    // GET method
    public async get<T>(url: string, params?: object): Promise<ApiResponse<T>> {
        try {
            return await this.axiosInstance.get<T>(url, { params });
        } catch (error) {
            throw error as ApiError;
        }
    }

    // POST method
    public async post<T>(url: string, data?: object): Promise<AxiosResponse<T>> {
        try {
            return await this.axiosInstance.post<T>(url, data);
        } catch (error) {
            throw error as ApiError;
        }
    }

    // PUT method
    public async put<T>(url: string, data?: object): Promise<ApiResponse<T>> {
        try {
            return await this.axiosInstance.put<T>(url, data);
        } catch (error) {
            throw error as ApiError;
        }
    }

    // DELETE method
    public async delete<T>(url: string): Promise<ApiResponse<T>> {
        try {
            return await this.axiosInstance.delete<T>(url);
        } catch (error) {
            throw error as ApiError;
        }
    }

    // File upload method
    public async uploadFile<T>(url: string, formData: FormData): Promise<AxiosResponse<T>> {
        try {
            return await this.axiosInstance.post<T>(url, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });
        } catch (error) {
            throw error as ApiError;
        }
    }

    /**
     * POST that returns the raw response body as an ArrayBuffer.
     * Used for binary responses like Kokoro TTS audio (audio/wav).
     * The axios `timeout` and auth interceptors still apply.
     */
    public async postBinary(
        url: string,
        data?: object,
        opts?: { timeoutMs?: number },
    ): Promise<ArrayBuffer> {
        try {
            const response = await this.axiosInstance.post<ArrayBuffer>(url, data, {
                responseType: 'arraybuffer',
                timeout: opts?.timeoutMs ?? 30000, // TTS can take a few seconds, especially on cold-start
            });
            return response.data;
        } catch (error) {
            throw error as ApiError;
        }
    }

    /**
     * Open an authenticated WebSocket to the backend.
     *
     * Same auth + addressing model as the REST methods: hits the same
     * baseURL host:port (just `ws://` instead of `http://`) and attaches
     * the JWT from localStorage. Browsers can't set custom headers on
     * `new WebSocket()`, so the token rides as a `?token=…` query param;
     * the backend verifies it at the upgrade boundary.
     */
    public openWebSocket(
        path: string,
        params?: Record<string, string | undefined>,
    ): WebSocket {
        const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = process.env.REACT_APP_BACKEND_SERVER_IP;
        const port = process.env.REACT_APP_BACKEND_PROXY_PORT;

        const qs = new URLSearchParams();
        if (params) {
            for (const [k, v] of Object.entries(params)) {
                if (v !== undefined && v !== '') qs.set(k, v);
            }
        }
        const token = localStorage.getItem('token');
        if (token) qs.set('token', token);

        const normalizedPath = path.startsWith('/') ? path : `/${path}`;
        const tail = qs.toString();
        return new WebSocket(
            `${wsProto}//${host}:${port}${normalizedPath}${tail ? `?${tail}` : ''}`,
        );
    }


    // Add other HTTP methods as needed (PUT, DELETE, etc.)
};

// Create an instance with your base URL
const apiService = new ApiService();

export default apiService;