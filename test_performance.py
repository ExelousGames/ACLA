"""
Performance and Load Testing for ACLA AI Service
"""

import requests
import time
import threading
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

BASE_URL = "http://localhost:8000"

class PerformanceTestRunner:
    """Performance and load testing for AI service"""
    
    def __init__(self):
        self.results = []
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict]:
        """Create sample data for testing"""
        return [
            {
                "Physics_speed_kmh": 120.5 + i,
                "Physics_rpm": 6500 + (i * 100),
                "Physics_gear": 4 + (i % 3),
                "Physics_gas": 0.8,
                "Physics_brake": 0.0,
                "Graphics_last_time": 94890 + (i * 100),
                "Graphics_current_time": 95250 + (i * 100),
                "Physics_fuel": 62 - i
            }
            for i in range(50)  # 50 data points
        ]
    
    def measure_response_time(self, url: str, method: str = "GET", data: Dict = None) -> Dict:
        """Measure response time for a single request"""
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=30)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "success": True,
                "response_time_ms": response_time,
                "status_code": response.status_code,
                "response_size": len(response.content)
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            return {
                "success": False,
                "response_time_ms": response_time,
                "error": str(e),
                "status_code": None,
                "response_size": 0
            }
    
    def test_health_check_performance(self, num_requests: int = 100) -> Dict:
        """Test health check endpoint performance"""
        print(f"🔥 Testing health check performance ({num_requests} requests)...")
        
        results = []
        
        for i in range(num_requests):
            result = self.measure_response_time(f"{BASE_URL}/health")
            results.append(result)
            
            if (i + 1) % 20 == 0:
                print(f"   Completed {i + 1}/{num_requests} requests")
        
        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["success"]]
        success_rate = len([r for r in results if r["success"]]) / len(results) * 100
        
        stats = {
            "endpoint": "/health",
            "total_requests": num_requests,
            "successful_requests": len(response_times),
            "success_rate_percent": success_rate,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "median_response_time_ms": statistics.median(response_times) if response_times else 0,
            "p95_response_time_ms": self._calculate_percentile(response_times, 95) if response_times else 0,
            "p99_response_time_ms": self._calculate_percentile(response_times, 99) if response_times else 0
        }
        
        return stats
    
    def test_concurrent_requests(self, num_threads: int = 10, requests_per_thread: int = 10) -> Dict:
        """Test concurrent request handling"""
        print(f"🔥 Testing concurrent requests ({num_threads} threads, {requests_per_thread} requests each)...")
        
        results = []
        
        def worker():
            thread_results = []
            for _ in range(requests_per_thread):
                result = self.measure_response_time(f"{BASE_URL}/health")
                thread_results.append(result)
            return thread_results
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                thread_results = future.result()
                results.extend(thread_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["success"]]
        success_rate = len([r for r in results if r["success"]]) / len(results) * 100
        
        total_requests = num_threads * requests_per_thread
        requests_per_second = total_requests / total_time
        
        stats = {
            "test_type": "concurrent_requests",
            "num_threads": num_threads,
            "requests_per_thread": requests_per_thread,
            "total_requests": total_requests,
            "total_time_seconds": total_time,
            "requests_per_second": requests_per_second,
            "successful_requests": len(response_times),
            "success_rate_percent": success_rate,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "p95_response_time_ms": self._calculate_percentile(response_times, 95) if response_times else 0
        }
        
        return stats
    
    def test_model_training_performance(self) -> Dict:
        """Test model training performance"""
        print("🔥 Testing model training performance...")
        
        payload = {
            "session_data": self.sample_data,
            "model_type": "lap_time_prediction",
            "training_parameters": {
                "n_estimators": 50
            }
        }
        
        result = self.measure_response_time(f"{BASE_URL}/model/train", "POST", payload)
        
        stats = {
            "endpoint": "/model/train",
            "data_points": len(self.sample_data),
            "model_type": "lap_time_prediction",
            "success": result["success"],
            "training_time_ms": result["response_time_ms"],
            "training_time_seconds": result["response_time_ms"] / 1000,
            "status_code": result.get("status_code"),
            "error": result.get("error")
        }
        
        return stats
    
    def test_dataset_upload_performance(self, data_sizes: List[int] = [10, 50, 100, 500]) -> List[Dict]:
        """Test dataset upload performance with different data sizes"""
        print("🔥 Testing dataset upload performance with different data sizes...")
        
        results = []
        
        for size in data_sizes:
            print(f"   Testing with {size} data points...")
            
            # Create data of specified size
            test_data = self.sample_data[:size] if size <= len(self.sample_data) else self.sample_data * (size // len(self.sample_data) + 1)
            test_data = test_data[:size]  # Ensure exact size
            
            payload = {
                "id": f"perf_test_{size}",
                "name": f"Performance Test {size} points",
                "data": test_data
            }
            
            result = self.measure_response_time(f"{BASE_URL}/datasets/upload", "POST", payload)
            
            stats = {
                "data_size": size,
                "success": result["success"],
                "upload_time_ms": result["response_time_ms"],
                "upload_time_seconds": result["response_time_ms"] / 1000,
                "throughput_points_per_second": size / (result["response_time_ms"] / 1000) if result["response_time_ms"] > 0 else 0,
                "response_size_bytes": result["response_size"],
                "status_code": result.get("status_code"),
                "error": result.get("error")
            }
            
            results.append(stats)
        
        return results
    
    def test_telemetry_upload_performance(self) -> Dict:
        """Test telemetry upload performance"""
        print("🔥 Testing telemetry upload performance...")
        
        # Convert session data to telemetry format
        telemetry_data = {}
        for key in self.sample_data[0].keys():
            telemetry_data[key] = [record[key] for record in self.sample_data]
        
        payload = {
            "session_id": "perf_test_telemetry",
            "telemetry_data": telemetry_data,
            "metadata": {
                "track": "brands_hatch",
                "car": "porsche_991ii_gt3_r"
            }
        }
        
        result = self.measure_response_time(f"{BASE_URL}/telemetry/upload", "POST", payload)
        
        stats = {
            "endpoint": "/telemetry/upload",
            "data_points": len(self.sample_data),
            "features": len(telemetry_data.keys()),
            "success": result["success"],
            "upload_time_ms": result["response_time_ms"],
            "upload_time_seconds": result["response_time_ms"] / 1000,
            "throughput_points_per_second": len(self.sample_data) / (result["response_time_ms"] / 1000) if result["response_time_ms"] > 0 else 0,
            "status_code": result.get("status_code"),
            "error": result.get("error")
        }
        
        return stats
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * len(sorted_data)
        
        if index.is_integer():
            return sorted_data[int(index) - 1]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1] if int(index) + 1 < len(sorted_data) else lower
            return lower + (upper - lower) * (index - int(index))
    
    def run_performance_suite(self) -> Dict:
        """Run complete performance test suite"""
        print("🚀 Starting Performance Test Suite")
        print("=" * 60)
        
        suite_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {}
        }
        
        # Test 1: Health check performance
        try:
            health_stats = self.test_health_check_performance(100)
            suite_results["tests"]["health_check"] = health_stats
            print(f"✅ Health check: {health_stats['avg_response_time_ms']:.2f}ms avg")
        except Exception as e:
            print(f"❌ Health check performance test failed: {str(e)}")
            suite_results["tests"]["health_check"] = {"error": str(e)}
        
        # Test 2: Concurrent requests
        try:
            concurrent_stats = self.test_concurrent_requests(10, 5)
            suite_results["tests"]["concurrent_requests"] = concurrent_stats
            print(f"✅ Concurrent requests: {concurrent_stats['requests_per_second']:.2f} req/sec")
        except Exception as e:
            print(f"❌ Concurrent requests test failed: {str(e)}")
            suite_results["tests"]["concurrent_requests"] = {"error": str(e)}
        
        # Test 3: Dataset upload performance
        try:
            upload_stats = self.test_dataset_upload_performance([10, 50, 100])
            suite_results["tests"]["dataset_upload"] = upload_stats
            print(f"✅ Dataset upload: tested sizes 10, 50, 100")
        except Exception as e:
            print(f"❌ Dataset upload performance test failed: {str(e)}")
            suite_results["tests"]["dataset_upload"] = {"error": str(e)}
        
        # Test 4: Model training performance
        try:
            training_stats = self.test_model_training_performance()
            suite_results["tests"]["model_training"] = training_stats
            print(f"✅ Model training: {training_stats['training_time_seconds']:.2f}s")
        except Exception as e:
            print(f"❌ Model training performance test failed: {str(e)}")
            suite_results["tests"]["model_training"] = {"error": str(e)}
        
        # Test 5: Telemetry upload performance
        try:
            telemetry_stats = self.test_telemetry_upload_performance()
            suite_results["tests"]["telemetry_upload"] = telemetry_stats
            print(f"✅ Telemetry upload: {telemetry_stats['upload_time_seconds']:.2f}s")
        except Exception as e:
            print(f"❌ Telemetry upload performance test failed: {str(e)}")
            suite_results["tests"]["telemetry_upload"] = {"error": str(e)}
        
        print("\n" + "=" * 60)
        print("Performance Test Suite Completed!")
        
        return suite_results
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate a formatted performance report"""
        report = []
        report.append("ACLA AI Service Performance Test Report")
        report.append("=" * 50)
        report.append(f"Test Date: {results['timestamp']}")
        report.append("")
        
        for test_name, test_results in results["tests"].items():
            report.append(f"📊 {test_name.replace('_', ' ').title()}")
            report.append("-" * 30)
            
            if "error" in test_results:
                report.append(f"❌ Error: {test_results['error']}")
            else:
                if test_name == "health_check":
                    report.append(f"   Average Response Time: {test_results['avg_response_time_ms']:.2f}ms")
                    report.append(f"   95th Percentile: {test_results['p95_response_time_ms']:.2f}ms")
                    report.append(f"   Success Rate: {test_results['success_rate_percent']:.1f}%")
                
                elif test_name == "concurrent_requests":
                    report.append(f"   Requests per Second: {test_results['requests_per_second']:.2f}")
                    report.append(f"   Average Response Time: {test_results['avg_response_time_ms']:.2f}ms")
                    report.append(f"   Success Rate: {test_results['success_rate_percent']:.1f}%")
                
                elif test_name == "dataset_upload":
                    for upload_test in test_results:
                        size = upload_test['data_size']
                        time_s = upload_test['upload_time_seconds']
                        throughput = upload_test['throughput_points_per_second']
                        report.append(f"   {size} points: {time_s:.2f}s ({throughput:.1f} points/sec)")
                
                elif test_name == "model_training":
                    report.append(f"   Training Time: {test_results['training_time_seconds']:.2f}s")
                    report.append(f"   Data Points: {test_results['data_points']}")
                    report.append(f"   Model Type: {test_results['model_type']}")
                
                elif test_name == "telemetry_upload":
                    report.append(f"   Upload Time: {test_results['upload_time_seconds']:.2f}s")
                    report.append(f"   Throughput: {test_results['throughput_points_per_second']:.1f} points/sec")
                    report.append(f"   Features: {test_results['features']}")
            
            report.append("")
        
        return "\n".join(report)

def run_load_test(duration_seconds: int = 60, concurrent_users: int = 5):
    """Run a sustained load test"""
    print(f"🔥 Running load test for {duration_seconds} seconds with {concurrent_users} concurrent users...")
    
    results = []
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    def user_simulation():
        user_results = []
        while time.time() < end_time:
            # Simulate user behavior
            try:
                # Health check
                response = requests.get(f"{BASE_URL}/health", timeout=10)
                user_results.append({
                    "endpoint": "/health",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                })
                
                time.sleep(1)  # Wait between requests
                
            except Exception as e:
                user_results.append({
                    "endpoint": "/health",
                    "error": str(e),
                    "status_code": None
                })
        
        return user_results
    
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(user_simulation) for _ in range(concurrent_users)]
        
        for future in as_completed(futures):
            user_results = future.result()
            results.extend(user_results)
    
    # Analyze results
    successful_requests = [r for r in results if r.get("status_code") == 200]
    total_requests = len(results)
    success_rate = len(successful_requests) / total_requests * 100 if total_requests > 0 else 0
    
    if successful_requests:
        response_times = [r["response_time_ms"] for r in successful_requests]
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
    else:
        avg_response_time = 0
        max_response_time = 0
    
    actual_duration = time.time() - start_time
    requests_per_second = total_requests / actual_duration
    
    print(f"✅ Load test completed!")
    print(f"   Duration: {actual_duration:.1f}s")
    print(f"   Total Requests: {total_requests}")
    print(f"   Requests per Second: {requests_per_second:.2f}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Average Response Time: {avg_response_time:.2f}ms")
    print(f"   Max Response Time: {max_response_time:.2f}ms")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        # Run load test
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        users = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        run_load_test(duration, users)
    else:
        # Run performance test suite
        runner = PerformanceTestRunner()
        results = runner.run_performance_suite()
        
        # Generate and save report
        report = runner.generate_performance_report(results)
        print("\n" + report)
        
        # Save results to file
        with open("performance_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        with open("performance_test_report.txt", "w") as f:
            f.write(report)
        
        print("\n📄 Results saved to performance_test_results.json and performance_test_report.txt")
