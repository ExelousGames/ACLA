import apiService from 'services/api.service';

export interface OpportunityForecastRequest {
    telemetry_data: Record<string, any>[];
    horizon_seconds?: number;
    top_k?: number;
}

export interface OpportunityForecastOpportunity {
    label_id: string;
    label_name: string;
    parent_label: string;
    probability: number;
    circuit_section_id?: string;
    circuit_section_name?: string;
}

export interface OpportunityForecastResponse {
    status: string;
    model_status?: string;
    horizon_seconds: number;
    opportunities: OpportunityForecastOpportunity[];
    circuit_section_match?: any;
}

export const getOpportunityForecast = async (
    request: OpportunityForecastRequest
): Promise<OpportunityForecastResponse> => {
    const response = await apiService.post<OpportunityForecastResponse>(
        '/racing-session/opportunity-forecast',
        request
    );
    return response.data;
};
