"""
Simple script to test the Dental X-Ray API endpoints.
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_api():
    print("🧪 Testing Dental X-Ray API...")
    print(f"API URL: {API_URL}")
    
    try:
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{API_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test model info endpoint
        print("\n3. Testing model info endpoint...")
        response = requests.get(f"{API_URL}/model_info")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
        
        # Test data stats endpoint
        print("\n4. Testing data stats endpoint...")
        response = requests.get(f"{API_URL}/data_stats")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        print("\n✅ API is working! You can:")
        print(f"   - View API docs: {API_URL}/docs")
        print(f"   - Test endpoints: {API_URL}/health")
        print(f"   - Upload images for prediction")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on port 8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_api()