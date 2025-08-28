"""
Simple test runner for ACLA AI Service APIs (no pytest dependency)
"""

import requests
import json
import time
import traceback
from typing import Dict, List, Any

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

class SimpleTestRunner:
    """Simple test runner for AI service APIs"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.sample_session_data = self._create_sample_session_data()
        self.sample_telemetry_data = self._create_sample_telemetry_data()
        self.test_dataset_id = None
        self.test_session_id = "test_session_001"
    
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
    
    def _create_sample_telemetry_data(self) -> Dict[str, List]:
        """Create sample telemetry data for testing"""
        return {
            "Physics_speed_kmh": [120.5, 118.2, 125.8, 115.3, 130.1],
            "Physics_rpm": [6500, 6200, 7000, 5800, 7200],
            "Physics_gear": [4, 4, 5, 3, 5],
            "Physics_gas": [0.8, 0.6, 1.0, 0.5, 0.9],
            "Physics_brake": [0.0, 0.2, 0.0, 0.4, 0.0],
            "Graphics_last_time": [94890, 95120, 93450, 96200, 92800],
            "Graphics_current_time": [95250, 95480, 93810, 96560, 93160],
            "Physics_fuel": [62, 61, 60, 59, 58]
        }
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and track results"""
        try:
            print(f"🧪 Running {test_name}...")
            test_func()
            print(f"✅ {test_name} PASSED")
            self.passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {str(e)}")
            self.failed += 1
            self.errors.append(f"{test_name}: {str(e)}")
    
    def assert_equal(self, actual, expected, message=""):
        """Simple assertion helper"""
        if actual != expected:
            raise AssertionError(f"{message}: Expected {expected}, got {actual}")
    
    def assert_in(self, item, container, message=""):
        """Assert item is in container"""
        if item not in container:
            raise AssertionError(f"{message}: {item} not found in {container}")
    
    def assert_status_code(self, response, expected_code):
        """Assert HTTP status code"""
        if response.status_code != expected_code:
            raise AssertionError(f"Expected status {expected_code}, got {response.status_code}. Response: {response.text}")
    
    # Test Methods
    def test_health_check(self):
        """Test the health check endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["status"], "healthy")
        self.assert_equal(data["service"], "ACLA AI Service")
    
    def test_upload_dataset(self):
        """Test dataset upload functionality"""
        payload = {
            "id": "test_dataset_001",
            "name": "Test Racing Dataset",
            "data": self.sample_session_data
        }
        
        response = requests.post(f"{BASE_URL}/datasets/upload", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["dataset_id"], "test_dataset_001")
        self.assert_in("metadata", data)
        self.assert_in("initial_analysis", data)
        
        # Store for later tests
        self.test_dataset_id = data["dataset_id"]
    
    def test_list_datasets(self):
        """Test listing datasets"""
        # Ensure we have a dataset first
        if not self.test_dataset_id:
            self.test_upload_dataset()
        
        response = requests.get(f"{BASE_URL}/datasets", timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_in("datasets", data)
    
    def test_analyze_dataset_basic_stats(self):
        """Test basic statistics analysis"""
        if not self.test_dataset_id:
            self.test_upload_dataset()
        
        payload = {
            "dataset_id": self.test_dataset_id,
            "analysis_type": "basic_stats"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["analysis_type"], "basic_stats")
        self.assert_in("result", data)
    
    def test_analyze_dataset_performance(self):
        """Test performance analysis"""
        if not self.test_dataset_id:
            self.test_upload_dataset()
        
        payload = {
            "dataset_id": self.test_dataset_id,
            "analysis_type": "performance"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["analysis_type"], "performance")
        self.assert_in("result", data)
    
    def test_process_query(self):
        """Test natural language query processing"""
        if not self.test_dataset_id:
            self.test_upload_dataset()
        
        payload = {
            "question": "What are the basic statistics?",
            "dataset_id": self.test_dataset_id
        }
        
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_in("answer", data)
    
    def test_analyze_racing_session(self):
        """Test racing session analysis"""
        payload = {
            "session_id": "test_session_race",
            "session_name": "Test Racing Session",
            "session_data": self.sample_session_data
        }
        
        response = requests.post(f"{BASE_URL}/racing-session/analyze", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["session_id"], "test_session_race")
        self.assert_in("analysis", data)
    
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
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["session_id"], self.test_session_id)
        self.assert_in("feature_validation", data)
        self.assert_in("telemetry_summary", data)
    
    def test_analyze_telemetry(self):
        """Test telemetry analysis"""
        # Ensure telemetry is uploaded first
        self.test_upload_telemetry_data()
        
        payload = {
            "session_id": self.test_session_id,
            "analysis_type": "comprehensive"
        }
        
        response = requests.post(f"{BASE_URL}/telemetry/analyze", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["session_id"], self.test_session_id)
        self.assert_equal(data["analysis_type"], "comprehensive")
    
    def test_get_telemetry_features(self):
        """Test telemetry features endpoint"""
        response = requests.get(f"{BASE_URL}/telemetry/features", timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_in("total_features", data)
        self.assert_in("feature_categories", data)
    
    def test_validate_telemetry_data(self):
        """Test telemetry data validation"""
        response = requests.post(f"{BASE_URL}/telemetry/validate", 
                               json=self.sample_telemetry_data, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_in("validation_result", data)
        self.assert_in("data_quality", data)
    
    def test_train_lap_time_model(self):
        """Test lap time prediction model training"""
        payload = {
            "session_data": self.sample_session_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {
                "n_estimators": 10
            }
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["status"], "success")
        self.assert_in("model_data", data)
        self.assert_in("performance_metrics", data)
    
    def test_train_sector_model(self):
        """Test sector analysis model training"""
        payload = {
            "session_data": self.sample_session_data,
            "model_type": "sector_analysis",
            "training_parameters": {
                "n_clusters": 3
            }
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["status"], "success")
    
    def test_train_setup_model(self):
        """Test setup optimization model training"""
        payload = {
            "session_data": self.sample_session_data,
            "model_type": "setup_optimization",
            "training_parameters": {
                "n_estimators": 5
            }
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=payload, timeout=60)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["status"], "success")
    
    def test_model_prediction(self):
        """Test model prediction"""
        # First train a model
        train_payload = {
            "session_data": self.sample_session_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {"n_estimators": 5}
        }
        
        train_response = requests.post(f"{BASE_URL}/model/train", json=train_payload, timeout=60)
        self.assert_status_code(train_response, 200)
        
        train_data = train_response.json()
        model_data = train_data["model_data"]
        model_metadata = train_data["model_metadata"]
        
        # Test prediction
        predict_payload = {
            "modelData": model_data,
            "modelMetadata": model_metadata,
            "inputData": [self.sample_session_data[0]]
        }
        
        response = requests.post(f"{BASE_URL}/model/predict", json=predict_payload, timeout=TIMEOUT)
        self.assert_status_code(response, 200)
        
        data = response.json()
        self.assert_equal(data["status"], "success")
        self.assert_in("predictions", data)
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test invalid dataset ID
        payload = {
            "dataset_id": "nonexistent",
            "analysis_type": "basic_stats"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 404)
        
        # Test invalid analysis type
        if not self.test_dataset_id:
            self.test_upload_dataset()
        
        payload = {
            "dataset_id": self.test_dataset_id,
            "analysis_type": "invalid_type"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        self.assert_status_code(response, 400)
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting ACLA AI Service API Tests")
        print("=" * 50)
        
        # Basic functionality tests
        self.run_test("Health Check", self.test_health_check)
        self.run_test("Upload Dataset", self.test_upload_dataset)
        self.run_test("List Datasets", self.test_list_datasets)
        
        # Analysis tests
        self.run_test("Analyze Dataset - Basic Stats", self.test_analyze_dataset_basic_stats)
        self.run_test("Analyze Dataset - Performance", self.test_analyze_dataset_performance)
        self.run_test("Process Query", self.test_process_query)
        
        # Racing session tests
        self.run_test("Analyze Racing Session", self.test_analyze_racing_session)
        
        # Telemetry tests
        self.run_test("Upload Telemetry Data", self.test_upload_telemetry_data)
        self.run_test("Analyze Telemetry", self.test_analyze_telemetry)
        self.run_test("Get Telemetry Features", self.test_get_telemetry_features)
        self.run_test("Validate Telemetry Data", self.test_validate_telemetry_data)
        
        # Model training tests
        self.run_test("Train Lap Time Model", self.test_train_lap_time_model)
        self.run_test("Train Sector Model", self.test_train_sector_model)
        self.run_test("Train Setup Model", self.test_train_setup_model)
        
        # Advanced tests
        self.run_test("Model Prediction", self.test_model_prediction)
        self.run_test("Error Handling", self.test_error_handling)
        
        # Print results
        print("\n" + "=" * 50)
        print(f"🎯 Test Results: {self.passed} PASSED, {self.failed} FAILED")
        
        if self.failed > 0:
            print("\n❌ Failed Tests:")
            for error in self.errors:
                print(f"   - {error}")
        else:
            print("\n🎉 All tests passed successfully!")
        
        return self.failed == 0

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("🔥 Running Quick Smoke Test...")
    
    try:
        # Test health check
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        print("✅ Health check passed")
        
        # Test basic dataset upload
        sample_data = [
            {"Physics_speed_kmh": 120, "Physics_rpm": 6500, "Graphics_last_time": 95000},
            {"Physics_speed_kmh": 115, "Physics_rpm": 6200, "Graphics_last_time": 96000}
        ]
        
        payload = {"id": "smoke_test", "name": "Smoke Test", "data": sample_data}
        response = requests.post(f"{BASE_URL}/datasets/upload", json=payload, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Dataset upload failed: {response.status_code}")
            return False
        
        print("✅ Dataset upload passed")
        
        # Test model training
        train_payload = {
            "session_data": sample_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {"n_estimators": 3}
        }
        
        response = requests.post(f"{BASE_URL}/model/train", json=train_payload, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Model training failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
        
        print("✅ Model training passed")
        print("🎉 Smoke test completed successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to AI service. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Smoke test failed: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Run quick smoke test
        success = run_quick_smoke_test()
        sys.exit(0 if success else 1)
    else:
        # Run full test suite
        runner = SimpleTestRunner()
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)
