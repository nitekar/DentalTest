"""
Locust load testing script for Dental X-Ray Classification API.
Tests prediction endpoint performance under various load conditions.
"""

from locust import HttpUser, task, between
import random
from pathlib import Path
import io
from PIL import Image
import numpy as np


class DentalAPIUser(HttpUser):
    """Simulates a user interacting with the dental classification API."""
    
    # Wait 1-3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts. Create test images."""
        self.test_images = self._generate_test_images()
    
    def _generate_test_images(self):
        """Generate synthetic test images for load testing."""
        images = []
        
        # Generate 10 random grayscale X-ray-like images
        for i in range(10):
            # Create random noise image (simulating X-ray)
            img_array = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            
            # Add some structure (simulating dental features)
            center = (112, 112)
            y, x = np.ogrid[-center[0]:224-center[0], -center[1]:224-center[1]]
            mask = x*x + y*y <= 80*80
            img_array[mask] = np.clip(img_array[mask] + 50, 0, 255)
            
            # Convert to PIL Image
            img = Image.fromarray(img_array, mode='L')
            img = img.convert('RGB')
            
            # Save to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            images.append(img_bytes)
        
        return images
    
    @task(10)
    def predict_single_image(self):
        """Test single image prediction endpoint (most common operation)."""
        # Select random test image
        img_bytes = random.choice(self.test_images)
        img_bytes.seek(0)
        
        # Prepare file upload
        files = {
            'file': ('test_xray.jpg', img_bytes, 'image/jpeg')
        }
        
        # Make prediction request
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if 'prediction' in result and 'confidence' in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure("API unhealthy")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def get_root(self):
        """Test root endpoint."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class StressTestUser(HttpUser):
    """High-frequency user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    def on_start(self):
        """Create test images."""
        self.test_images = self._generate_test_images()
    
    def _generate_test_images(self):
        """Generate test images."""
        images = []
        for i in range(5):
            img_array = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L').convert('RGB')
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            
            images.append(img_bytes)
        
        return images
    
    @task
    def rapid_predictions(self):
        """Rapid-fire predictions for stress testing."""
        img_bytes = random.choice(self.test_images)
        img_bytes.seek(0)
        
        files = {'file': ('xray.jpg', img_bytes, 'image/jpeg')}
        
        self.client.post("/predict", files=files)


# Custom configuration for different test scenarios
class LightLoadUser(HttpUser):
    """Light load user - typical production traffic."""
    
    wait_time = between(5, 10)
    weight = 3
    
    def on_start(self):
        self.test_images = [self._create_test_image()]
    
    def _create_test_image(self):
        img = Image.new('RGB', (224, 224), color='gray')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    @task
    def predict(self):
        img_bytes = self.test_images[0]
        img_bytes.seek(0)
        
        files = {'file': ('xray.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)


class HeavyLoadUser(HttpUser):
    """Heavy load user - peak traffic simulation."""
    
    wait_time = between(0.5, 2)
    weight = 1
    
    def on_start(self):
        self.test_images = [self._create_test_image() for _ in range(3)]
    
    def _create_test_image(self):
        img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    @task(5)
    def predict(self):
        img_bytes = random.choice(self.test_images)
        img_bytes.seek(0)
        
        files = {'file': ('xray.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)
    
    @task(1)
    def health(self):
        self.client.get("/health")


"""
USAGE INSTRUCTIONS:

1. Basic Load Test (Web UI):
   locust -f locustfile.py --host=http://localhost:8000
   Open http://localhost:8089
   Set users and spawn rate

2. Headless Mode (Command Line):
   locust -f locustfile.py --host=http://localhost:8000 \
          --users 100 --spawn-rate 10 --run-time 5m --headless

3. Test Different Container Scales:
   
   # 1 container
   docker-compose up --scale api=1
   locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 3m --headless --csv=results_1
   
   # 2 containers
   docker-compose up --scale api=2
   locust -f locustfile.py --host=http://localhost:8000 --users 250 --spawn-rate 25 --run-time 3m --headless --csv=results_2
   
   # 4 containers
   docker-compose up --scale api=4
   locust -f locustfile.py --host=http://localhost:8000 --users 500 --spawn-rate 50 --run-time 3m --headless --csv=results_4

4. Stress Test:
   locust -f locustfile.py --host=http://localhost:8000 \
          --users 1000 --spawn-rate 100 --run-time 10m \
          --headless --csv=stress_test

5. Custom User Mix:
   locust -f locustfile.py --host=http://localhost:8000 \
          DentalAPIUser LightLoadUser HeavyLoadUser

METRICS TO MONITOR:
- Request Count
- Failures
- Median Response Time
- 95th Percentile
- 99th Percentile
- Requests per Second (RPS)
- Average Response Size
"""


if __name__ == "__main__":
    print("Load testing script for Dental X-Ray Classification API")
    print("\nUsage:")
    print("  locust -f locustfile.py --host=http://localhost:8000")
    print("\nOr headless:")
    print("  locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless")