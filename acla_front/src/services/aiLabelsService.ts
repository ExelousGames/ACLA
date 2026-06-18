import apiService from 'services/api.service';

export interface AiLabelsResponse {
    label_mapping: Record<string, string>;
    label_name_to_id: Record<string, string>;
    label_image_map: Record<string, string>;
    label_categories: Record<string, string[]>;
}

export const getAiLabels = async (): Promise<AiLabelsResponse> => {
    const response = await apiService.get<AiLabelsResponse>('/racing-session/labels');
    return response.data;
};
