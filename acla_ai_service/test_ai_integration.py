"""
Test script for ACLA AI Service OpenAI Integration
"""

import asyncio
import httpx
import json
from typing import Dict, Any


class ACLAAITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
    
    async def test_health(self):
        """Test service health"""
        print("🔍 Testing service health...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                print(f"✅ Health check: {response.status_code}")
                return response.json()
            except Exception as e:
                print(f"❌ Health check failed: {e}")
                return None
    
    async def test_ai_capabilities(self):
        """Test AI capabilities endpoint"""
        print("🧠 Testing AI capabilities...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/v1/ai/capabilities")
                result = response.json()
                print(f"✅ AI Capabilities: OpenAI configured = {result.get('openai_configured')}")
                print(f"   Available functions: {len(result.get('available_functions', []))}")
                return result
            except Exception as e:
                print(f"❌ AI capabilities test failed: {e}")
                return None
    
    async def upload_sample_data(self):
        """Upload sample racing session data"""
        print("📤 Uploading sample racing data...")
        
        sample_data = {
            "id": "test_session_001",
            "name": "Test Racing Session - Spa",
            "data": [
                {
                    "timestamp": 0.0,
                    "lap": 1,
                    "speed": 180.5,
                    "throttle": 0.85,
                    "brake": 0.0,
                    "steering": 0.1,
                    "gear": 6,
                    "track_position": 0.1
                },
                {
                    "timestamp": 1.0,
                    "lap": 1,
                    "speed": 185.2,
                    "throttle": 0.90,
                    "brake": 0.0,
                    "steering": 0.05,
                    "gear": 6,
                    "track_position": 0.12
                },
                {
                    "timestamp": 2.0,
                    "lap": 1,
                    "speed": 190.0,
                    "throttle": 1.0,
                    "brake": 0.0,
                    "steering": 0.0,
                    "gear": 6,
                    "track_position": 0.15
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/datasets/upload",
                    json=sample_data
                )
                if response.status_code == 200:
                    print("✅ Sample data uploaded successfully")
                    self.session_id = "test_session_001"
                    return response.json()
                else:
                    print(f"❌ Data upload failed: {response.status_code}")
                    return None
            except Exception as e:
                print(f"❌ Data upload failed: {e}")
                return None
    
    async def test_natural_language_queries(self):
        """Test natural language query processing"""
        if not self.session_id:
            print("⚠️ No session data available for queries")
            return
        
        test_queries = [
            "What was my fastest lap time?",
            "How can I improve my cornering?",
            "Analyze my performance in this session",
            "Compare my throttle and brake usage",
            "Give me coaching advice for better consistency",
            "Train a lap time prediction model using this session data"
        ]
        
        print("🗣️ Testing natural language queries...")
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            
            request_data = {
                "question": query,
                "dataset_id": self.session_id,
                "user_id": "test_user_001",
                "context": {
                    "type": "performance_query",
                    "skill_level": "intermediate"
                }
            }
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/query",
                        json=request_data,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Answer: {result.get('answer', 'No answer provided')}")
                        
                        function_calls = result.get('function_calls', [])
                        if function_calls:
                            print(f"🔧 Functions called: {[fc.get('function') for fc in function_calls]}")
                        
                        if result.get('error'):
                            print(f"⚠️ Error: {result.get('error')}")
                            
                    else:
                        print(f"❌ Query failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Query failed: {e}")
    
    async def test_ai_conversation(self):
        """Test AI conversation endpoint"""
        if not self.session_id:
            print("⚠️ No session data available for conversation")
            return
        
        print("💬 Testing AI conversation...")
        
        conversation_request = {
            "question": "I'm a beginner driver. Can you help me understand what I should focus on to improve my lap times?",
            "dataset_id": self.session_id,
            "user_id": "test_user_001",
            "context": {
                "skill_level": "beginner",
                "track": "spa-francorchamps"
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/ai/conversation",
                    json=conversation_request,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ AI Response: {result.get('response')}")
                else:
                    print(f"❌ Conversation failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Conversation failed: {e}")
    
    async def test_coaching_advice(self):
        """Test coaching advice generation"""
        if not self.session_id:
            print("⚠️ No session data available for coaching")
            return
        
        print("🏁 Testing coaching advice...")
        
        coaching_request = {
            "question": "Give me personalized coaching advice",
            "dataset_id": self.session_id,
            "user_id": "test_user_001",
            "context": {
                "skill_level": "intermediate",
                "focus_area": "consistency"
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/ai/coach-advice",
                    json=coaching_request,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Coaching Advice: {result.get('coaching_advice')}")
                else:
                    print(f"❌ Coaching advice failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Coaching advice failed: {e}")
    
    async def test_backend_function_discovery(self):
        """Test if AI can discover and call backend functions"""
        print("🔧 Testing backend function discovery...")
        
        function_test_query = {
            "question": "Show me all my AI models for Spa track",
            "user_id": "test_user_001",
            "context": {
                "type": "model_query",
                "track_name": "spa-francorchamps"
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query",
                    json=function_test_query,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Backend function test result: {result.get('answer')}")
                    
                    function_calls = result.get('function_calls', [])
                    if function_calls:
                        backend_calls = [fc for fc in function_calls if fc.get('function') == 'call_backend_function']
                        if backend_calls:
                            print(f"🎯 Backend function called successfully: {len(backend_calls)} calls")
                        else:
                            print("ℹ️ No backend functions called for this query")
                    
                else:
                    print(f"❌ Backend function test failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Backend function test failed: {e}")
    
    async def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🚀 Starting ACLA AI Service Test Suite...")
        print("=" * 50)
        
        # Test 1: Health check
        health_result = await self.test_health()
        if not health_result:
            print("❌ Service not available, stopping tests")
            return
        
        # Test 2: AI capabilities
        capabilities = await self.test_ai_capabilities()
        
        # Test 3: Upload sample data
        await self.upload_sample_data()
        
        # Test 4: Natural language queries
        await self.test_natural_language_queries()
        
        # Test 5: AI conversation
        await self.test_ai_conversation()
        
        # Test 6: Coaching advice
        await self.test_coaching_advice()
        
        # Test 7: Backend function discovery
        await self.test_backend_function_discovery()
        
        print("\n" + "=" * 50)
        print("🏁 Test suite completed!")
        
        if capabilities and capabilities.get('openai_configured'):
            print("✅ OpenAI integration is working!")
        else:
            print("⚠️ OpenAI integration not configured - using fallback mode")


async def main():
    """Main test function"""
    tester = ACLAAITester()
    await tester.run_full_test_suite()


if __name__ == "__main__":
    asyncio.run(main())
