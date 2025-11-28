"""
Test script for verifying retraining functionality.
"""
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

def test_retrain_workflow():
    """Test the complete retraining workflow."""
    
    print("🧪 Testing Retraining Workflow\n")
    
    # 1. Check API health
    print("1️⃣ Checking API health...")
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ API healthy: model_loaded={data['model_loaded']}")
    else:
        print(f"   ❌ API health check failed: {response.status_code}")
        return
    
    # 2. Check current retrain status
    print("\n2️⃣ Checking retrain status...")
    response = requests.get(f"{API_URL}/retrain_status")
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Can retrain: {data['can_retrain']}")
        print(f"   New data available: {data['new_data_available']}")
    
    # 3. Check if new data is available
    new_data_dir = Path("data/new")
    has_new_data = new_data_dir.exists() and any(new_data_dir.iterdir())
    
    if not has_new_data:
        print("\n⚠️  No new training data found in data/new/")
        print("   To test retraining:")
        print("   1. Upload images via /upload_bulk endpoint")
        print("   2. Or manually create data/new/class_name/ folders with images")
        return
    
    # 4. Get data statistics before retraining
    print("\n3️⃣ Checking training data statistics...")
    response = requests.get(f"{API_URL}/data_stats")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total images: {data['total_images']}")
        print(f"   Classes: {data['num_classes']}")
        print(f"   Distribution: {data['class_distribution']}")
    
    # 5. Trigger retraining
    print("\n4️⃣ Triggering retraining (5 epochs for quick test)...")
    response = requests.post(
        f"{API_URL}/retrain",
        params={"epochs": 5, "learning_rate": 0.0001}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ {data['message']}")
        print(f"   Check status at: {data['check_status_at']}")
    elif response.status_code == 409:
        print(f"   ⚠️  Retraining already in progress")
        return
    elif response.status_code == 400:
        print(f"   ❌ {response.json()['detail']}")
        return
    else:
        print(f"   ❌ Failed to trigger retraining: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    # 6. Monitor retraining progress
    print("\n5️⃣ Monitoring retraining progress...\n")
    
    last_progress = -1
    while True:
        time.sleep(3)  # Check every 3 seconds
        
        response = requests.get(f"{API_URL}/retrain_status")
        if response.status_code == 200:
            data = response.json()
            status = data['status']
            progress = data['progress']
            message = data['message']
            
            # Only print if progress changed
            if progress != last_progress:
                print(f"   [{status.upper()}] {progress}% - {message}")
                last_progress = progress
            
            if status == "completed":
                print("\n   ✅ Retraining completed successfully!")
                break
            elif status == "failed":
                print(f"\n   ❌ Retraining failed: {message}")
                break
            elif status not in ["running", "idle"]:
                print(f"\n   ⚠️  Unknown status: {status}")
                break
    
    # 7. Verify model was updated
    print("\n6️⃣ Verifying model update...")
    response = requests.get(f"{API_URL}/model_info")
    if response.status_code == 200:
        data = response.json()
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Classes: {data['class_names']}")
        print(f"   Device: {data['device']}")
    
    # 8. Get updated data statistics
    print("\n7️⃣ Checking updated data statistics...")
    response = requests.get(f"{API_URL}/data_stats")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total images: {data['total_images']}")
        print(f"   Classes: {data['num_classes']}")
        print(f"   Distribution: {data['class_distribution']}")
    
    print("\n✅ Retraining workflow test completed!")


if __name__ == "__main__":
    try:
        test_retrain_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
