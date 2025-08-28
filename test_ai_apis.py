"""
Comprehensive test suite for ACLA AI Service APIs
"""

import pytest
import requests
import json
import time
from typing import Dict, List, Any

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

class TestACLAAIService:
    """Test suite for ACLA AI Service APIs"""
    
    @pytest.fixture(scope="session", autouse=True)
    def setup_test_data(self):
        """Setup test data that will be used across multiple tests"""
        self.sample_telemetry_data = self._create_sample_telemetry_data()
        self.sample_session_data = self._create_sample_session_data()
        self.test_dataset_id = None
        self.test_session_id = "test_session_001"
    
    def _create_sample_telemetry_data(self) -> Dict[str, List]:
        """Create sample telemetry data for testing"""
        data = {}
        sample_fields = [
            ("Physics_speed_kmh", [120.5, 118.2, 125.8, 115.3, 130.1]),
            ("Physics_rpm", [6500, 6200, 7000, 5800, 7200]),
            ("Physics_gear", [4, 4, 5, 3, 5]),
            ("Physics_gas", [0.8, 0.6, 1.0, 0.5, 0.9]),
            ("Physics_brake", [0.0, 0.2, 0.0, 0.4, 0.0]),
            ("Graphics_last_time", [94890, 95120, 93450, 96200, 92800]),
            ("Graphics_current_time", [95250, 95480, 93810, 96560, 93160]),
            ("Physics_fuel", [62, 61, 60, 59, 58])
        ]
        
        for field, values in sample_fields:
            data[field] = values
        
        return data
    
    def _create_sample_session_data(self) -> List[Dict]:
        """Create sample session data for testing"""
        return [
            {
                "Physics_speed_kmh": 120.5,
                "Physics_rpm": 6500,
                "Physics_gear": 4,
                "Physics_gas": 0.8,
                "Physics_brake": 0.0,
                "Graphics_last_time": 94890,
                "Graphics_current_time": 95250,
                "Physics_fuel": 62
            },
            {
                "Physics_speed_kmh": 118.2,
                "Physics_rpm": 6200,
                "Physics_gear": 4,
                "Physics_gas": 0.6,
                "Physics_brake": 0.2,
                "Graphics_last_time": 95120,
                "Graphics_current_time": 95480,
                "Physics_fuel": 61
            },
            {
                "Physics_speed_kmh": 125.8,
                "Physics_rpm": 7000,
                "Physics_gear": 5,
                "Physics_gas": 1.0,
                "Physics_brake": 0.0,
                "Graphics_last_time": 93450,
                "Graphics_current_time": 93810,
                "Physics_fuel": 60
            }
        ]

    # Health Check Tests
    def test_health_check(self):
        """Test the health check endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ACLA AI Service"

    # Dataset Management Tests
    def test_upload_dataset(self):
        """Test dataset upload functionality"""
        payload = {
            "id": "test_dataset_001",
            "name": "Test Racing Dataset",
            "data": self.sample_session_data
        }
        
        response = requests.post(f"{BASE_URL}/datasets/upload", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "test_dataset_001"
        assert data["message"] == "Dataset uploaded successfully"
        assert "metadata" in data
        assert "initial_analysis" in data
        
        # Store dataset ID for later tests
        self.test_dataset_id = data["dataset_id"]

    def test_upload_dataset_no_data(self):
        """Test dataset upload with no data (should fail)"""
        payload = {
            "id": "test_dataset_empty",
            "name": "Empty Dataset"
            # Missing "data" field
        }
        
        response = requests.post(f"{BASE_URL}/datasets/upload", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 400
        assert "No data provided" in response.text

    def test_list_datasets(self):
        """Test listing all datasets"""
        # First upload a dataset
        self.test_upload_dataset()
        
        response = requests.get(f"{BASE_URL}/datasets", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert len(data["datasets"]) > 0

    # Query Processing Tests
    def test_process_query_with_dataset(self):
        """Test natural language query processing"""
        # Ensure we have a dataset
        self.test_upload_dataset()
        
        payload = {
            "question": "What are the basic statistics of this dataset?",
            "dataset_id": self.test_dataset_id
        }
        
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "data" in data
        assert len(data["answer"]) > 0

    def test_process_query_without_dataset(self):
        """Test query processing without specifying a dataset"""
        payload = {
            "question": "How can I analyze my racing data?"
        }
        
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "suggested_actions" in data

    def test_process_query_performance(self):
        """Test performance-related query"""
        self.test_upload_dataset()
        
        payload = {
            "question": "Show me the performance analysis",
            "dataset_id": self.test_dataset_id
        }
        
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    # Analysis Tests
    def test_analyze_basic_stats(self):
        """Test basic statistics analysis"""
        self.test_upload_dataset()
        
        payload = {
            "dataset_id": self.test_dataset_id,
            "analysis_type": "basic_stats"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "basic_stats"
        assert "result" in data
        assert "shape" in data["result"]
        assert "columns" in data["result"]

    def test_analyze_correlation(self):
        """Test correlation analysis"""
        self.test_upload_dataset()
        
        payload = {
            "dataset_id": self.test_dataset_id,
            "analysis_type": "correlation"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "correlation"
        assert "result" in data

    def test_analyze_performance(self):
        """Test performance analysis"""
        self.test_upload_dataset()
        
        payload = {
            "dataset_id": self.test_dataset_id,
            "analysis_type": "performance"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "performance"
        assert "result" in data

    def test_analyze_invalid_type(self):
        """Test analysis with invalid type"""
        self.test_upload_dataset()
        
        payload = {
            "dataset_id": self.test_dataset_id,
            "analysis_type": "invalid_analysis"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 400
        assert "Unsupported analysis type" in response.text

    def test_analyze_nonexistent_dataset(self):
        """Test analysis with non-existent dataset"""
        payload = {
            "dataset_id": "nonexistent_dataset",
            "analysis_type": "basic_stats"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 404
        assert "Dataset not found" in response.text

    # Racing Session Analysis Tests
    def test_analyze_racing_session(self):
        """Test racing session analysis"""
        payload = {
            "session_id": "test_session_001",
            "session_name": "Test Session",
            "session_data": self.sample_session_data
        }
        
        response = requests.post(f"{BASE_URL}/racing-session/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert "analysis" in data
        assert "suggestions" in data

    def test_detect_racing_patterns(self):
        """Test racing pattern detection"""
        # First analyze a session to ensure it exists
        self.test_analyze_racing_session()
        
        payload = {
            "session_id": "test_session_001"
        }
        
        response = requests.post(f"{BASE_URL}/racing-session/patterns", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert "patterns" in data

    def test_get_performance_score(self):
        """Test performance score calculation"""
        self.test_analyze_racing_session()
        
        payload = {
            "session_id": "test_session_001"
        }
        
        response = requests.post(f"{BASE_URL}/racing-session/performance-score", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert "performance_score" in data

    def test_get_sector_analysis(self):
        """Test sector analysis"""
        self.test_analyze_racing_session()
        
        payload = {
            "session_id": "test_session_001"
        }
        
        response = requests.post(f"{BASE_URL}/racing-session/sector-analysis", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert "sector_analysis" in data

    def test_predict_optimal_lap_time(self):
        """Test optimal lap time prediction"""
        self.test_analyze_racing_session()
        
        payload = {
            "session_id": "test_session_001"
        }
        
        response = requests.post(f"{BASE_URL}/racing-session/optimal-prediction", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert "optimal_prediction" in data

    # Telemetry Tests
    def test_upload_telemetry_data(self):
        """Test telemetry data upload"""
        payload = {
            "session_id": self.test_session_id,
            "telemetry_data": self.sample_telemetry_data,
            "metadata": {
                "track": "brands_hatch",
                "car": "porsche_991ii_gt3_r"
            }
        }
        
        response = requests.post(f"{BASE_URL}/telemetry/upload", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == self.test_session_id
        assert "feature_validation" in data
        assert "telemetry_summary" in data
        assert "available_analyses" in data

    def test_analyze_telemetry_comprehensive(self):
        """Test comprehensive telemetry analysis"""
        # First upload telemetry data
        self.test_upload_telemetry_data()
        
        payload = {
            "session_id": self.test_session_id,
            "analysis_type": "comprehensive"
        }
        
        response = requests.post(f"{BASE_URL}/telemetry/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == self.test_session_id
        assert data["analysis_type"] == "comprehensive"
        assert "result" in data

    def test_analyze_telemetry_performance(self):
        """Test performance telemetry analysis"""
        self.test_upload_telemetry_data()
        
        payload = {
            "session_id": self.test_session_id,
            "analysis_type": "performance"
        }
        
        response = requests.post(f"{BASE_URL}/telemetry/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "performance"

    def test_analyze_telemetry_setup(self):
        """Test setup telemetry analysis"""
        self.test_upload_telemetry_data()
        
        payload = {
            "session_id": self.test_session_id,
            "analysis_type": "setup"
        }
        
        response = requests.post(f"{BASE_URL}/telemetry/analyze", json=payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_type"] == "setup"

    def test_get_telemetry_features(self):
        """Test telemetry features endpoint"""
        response = requests.get(f"{BASE_URL}/telemetry/features", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_features" in data
        assert "feature_categories" in data
        assert "performance_critical" in data
        assert "setup_features" in data

    def test_validate_telemetry_data(self):
        """Test telemetry data validation"""
        response = requests.post(f"{BASE_URL}/telemetry/validate", 
                               json=self.sample_telemetry_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert "validation_result" in data
        assert "data_quality" in data
        assert "performance_metrics" in data
        assert "recommendations" in data

    # Model Training Tests
    def test_train_lap_time_prediction_model(self):
        """Test lap time prediction model training"""
        payload = {
            "session_data": self.sample_session_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {
                "n_estimators": 10  # Small for testing
            }
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "model_data" in data
        assert "performance_metrics" in data
        assert "model_metadata" in data
        assert data["model_metadata"]["model_type"] == "lap_time_prediction"

    def test_train_sector_analysis_model(self):
        """Test sector analysis model training"""
        payload = {
            "session_data": self.sample_session_data,
            "model_type": "sector_analysis",
            "training_parameters": {
                "n_clusters": 3
            }
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_metadata"]["model_type"] == "sector_analysis"

    def test_train_setup_optimization_model(self):
        """Test setup optimization model training"""
        payload = {
            "session_data": self.sample_session_data,
            "model_type": "setup_optimization",
            "training_parameters": {
                "n_estimators": 5
            }
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_metadata"]["model_type"] == "setup_optimization"

    def test_train_model_invalid_type(self):
        """Test model training with invalid type"""
        payload = {
            "session_data": self.sample_session_data,
            "model_type": "invalid_type"
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        
        assert response.status_code == 400
        assert "Unsupported model type" in response.text

    def test_train_model_no_data(self):
        """Test model training with no data"""
        payload = {
            "session_data": [],
            "model_type": "lap_time_prediction"
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        
        assert response.status_code == 400
        assert "No training data provided" in response.text

    # Model Prediction Tests
    def test_model_prediction(self):
        """Test model prediction"""
        # First train a model
        train_payload = {
            "session_data": self.sample_session_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {"n_estimators": 5}
        }
        
        train_response = requests.post(f"{BASE_URL}/model/train", json=train_payload, timeout=60)
        assert train_response.status_code == 200
        
        model_data = train_response.json()["model_data"]
        model_metadata = train_response.json()["model_metadata"]
        
        # Test prediction
        predict_payload = {
            "modelData": model_data,
            "modelMetadata": model_metadata,
            "inputData": [self.sample_session_data[0]],
            "predictionOptions": {}
        }
        
        response = requests.post(f"{BASE_URL}/model/predict", json=predict_payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "predictions" in data

    def test_model_validation(self):
        """Test model validation"""
        # First train a model
        train_payload = {
            "session_data": self.sample_session_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {"n_estimators": 5}
        }
        
        train_response = requests.post(f"{BASE_URL}/model/train", json=train_payload, timeout=60)
        assert train_response.status_code == 200
        
        model_data = train_response.json()["model_data"]
        model_metadata = train_response.json()["model_metadata"]
        
        # Test validation
        validate_payload = {
            "modelData": model_data,
            "testData": self.sample_session_data,
            "modelMetadata": model_metadata
        }
        
        response = requests.post(f"{BASE_URL}/model/validate", json=validate_payload, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "validation_metrics" in data

    def test_get_model_metrics(self):
        """Test getting model metrics"""
        model_id = "test_model_001"
        
        response = requests.get(f"{BASE_URL}/model/metrics/{model_id}", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_id"] == model_id
        assert "metrics" in data

    # Error Handling Tests
    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint"""
        response = requests.get(f"{BASE_URL}/invalid-endpoint", timeout=TIMEOUT)
        
        assert response.status_code == 404

    def test_invalid_json_payload(self):
        """Test sending invalid JSON"""
        headers = {'Content-Type': 'application/json'}
        invalid_json = '{"invalid": json}'
        
        response = requests.post(f"{BASE_URL}/query", data=invalid_json, 
                               headers=headers, timeout=TIMEOUT)
        
        assert response.status_code == 422  # Unprocessable Entity

    # Performance Tests
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            return requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        for response in results:
            assert response.status_code == 200

    # Integration Tests
    def test_full_workflow(self):
        """Test complete workflow: upload -> analyze -> train -> predict"""
        # 1. Upload dataset
        upload_payload = {
            "id": "workflow_test",
            "name": "Workflow Test Dataset",
            "data": self.sample_session_data
        }
        
        upload_response = requests.post(f"{BASE_URL}/datasets/upload", 
                                      json=upload_payload, timeout=TIMEOUT)
        assert upload_response.status_code == 200
        dataset_id = upload_response.json()["dataset_id"]
        
        # 2. Analyze dataset
        analyze_payload = {
            "dataset_id": dataset_id,
            "analysis_type": "performance"
        }
        
        analyze_response = requests.post(f"{BASE_URL}/analyze", 
                                       json=analyze_payload, timeout=TIMEOUT)
        assert analyze_response.status_code == 200
        
        # 3. Train model
        train_payload = {
            "session_data": self.sample_session_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {"n_estimators": 5}
        }
        
        train_response = requests.post(f"{BASE_URL}/model/train", 
                                     json=train_payload, timeout=60)
        assert train_response.status_code == 200
        
        # 4. Make prediction
        model_data = train_response.json()["model_data"]
        model_metadata = train_response.json()["model_metadata"]
        
        predict_payload = {
            "modelData": model_data,
            "modelMetadata": model_metadata,
            "inputData": [self.sample_session_data[0]]
        }
        
        predict_response = requests.post(f"{BASE_URL}/model/predict", 
                                       json=predict_payload, timeout=TIMEOUT)
        assert predict_response.status_code == 200
        
        print("✅ Full workflow test completed successfully!")


if __name__ == "__main__":
    # Run specific tests manually for debugging
    test_instance = TestACLAAIService()
    test_instance.setup_test_data()
    
    print("Running manual test suite...")
    
    try:
        test_instance.test_health_check()
        print("✅ Health check passed")
        
        test_instance.test_upload_dataset()
        print("✅ Dataset upload passed")
        
        test_instance.test_train_lap_time_prediction_model()
        print("✅ Model training passed")
        
        test_instance.test_upload_telemetry_data()
        print("✅ Telemetry upload passed")
        
        print("\n🎉 All manual tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
