
import sys
import os
from fastapi.testclient import TestClient

# Add project root to sys.path to allow importing app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

client = TestClient(app)

def test_root():
    """Test the root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "endpoints" in data

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Status can be healthy or unhealthy depending on router initialization
    assert data["status"] in ["healthy", "unhealthy"]
    assert "version" in data

def test_simple_endpoint():
    """Test the simple test endpoint"""
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
