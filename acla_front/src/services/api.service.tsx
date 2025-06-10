import axios, { AxiosError, AxiosInstance, AxiosResponse } from 'axios';
import { useAuth } from 'hooks/AuthProvider';

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
    private userAuth: any;
    constructor() {
        this.axiosInstance = axios.create({
            baseURL: 'http://' + process.env.REACT_APP_BACKEND_SERVER_IP + ":" + process.env.REACT_APP_BACKEND_PROXY_PORT,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
            },
        });
        const userAuth = useAuth();
        this.setupInterceptors();
    }

    private setupInterceptors() {
        // Request interceptor
        this.axiosInstance.interceptors.request.use(
            (config) => {
                // Add auth token if exists
                const token = localStorage.getItem('token');
                if (token) {
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


    // Add other HTTP methods as needed (PUT, DELETE, etc.)
};