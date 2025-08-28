#!/usr/bin/env python3
"""
Test script for ACLA AI Service
"""

import requests
import json
import time

def test_ai_service():
    base_url = "http://localhost:8000"
    
    print("Testing ACLA AI Service...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test dataset upload
    test_data = {
        "name": "Test Racing Session",
        "data": [
            {"lap": 1, "time": 95.5, "speed": 120.3},
            {"lap": 2, "time": 94.2, "speed": 125.1},
            {"lap": 3, "time": 93.8, "speed": 127.5}
        ]
    }
    
    try:
        response = requests.post(f"{base_url}/datasets/upload", json=test_data)
        if response.status_code == 200:
            print("✅ Dataset upload passed")
            dataset_result = response.json()
            dataset_id = dataset_result.get("dataset_id")
            print(f"   Dataset ID: {dataset_id}")
        else:
            print("❌ Dataset upload failed")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Dataset upload failed: {e}")
        return False
    
    # Test query processing
    query_data = {
        "question": "What's the average lap time?",
        "dataset_id": dataset_id
    }
    
    try:
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            print("✅ Query processing passed")
            query_result = response.json()
            print(f"   Answer: {query_result.get('answer', 'No answer')}")
        else:
            print("❌ Query processing failed")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query processing failed: {e}")
        return False
    
    # Test analysis
    analysis_data = {
        "dataset_id": dataset_id,
        "analysis_type": "basic_stats"
    }
    
    try:
        response = requests.post(f"{base_url}/analyze", json=analysis_data)
        if response.status_code == 200:
            print("✅ Analysis passed")
            analysis_result = response.json()
            print(f"   Analysis type: {analysis_result.get('analysis_type')}")
        else:
            print("❌ Analysis failed")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    print("\n🎉 All tests passed! AI Service is working correctly.")
    return True

if __name__ == "__main__":
    test_ai_service()
