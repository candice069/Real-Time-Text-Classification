"""
Unit tests for the Text Classification API
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../api/src"))

from main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_ready_check(self):
        """Test the readiness check endpoint."""
        response = client.get("/ready")
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_predict_single_valid(self):
        """Test single prediction with valid input."""
        response = client.post(
            "/predict",
            json={"text": "Apple announces new iPhone with AI features"}
        )
        # May fail if model not loaded (503) or succeed
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "processing_time_ms" in data
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_single_empty(self):
        """Test single prediction with empty text."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422
    
    def test_predict_single_too_long(self):
        """Test single prediction with text exceeding max length."""
        long_text = "a" * 10001
        response = client.post(
            "/predict",
            json={"text": long_text}
        )
        assert response.status_code == 422
    
    def test_predict_batch_valid(self):
        """Test batch prediction with valid inputs."""
        texts = [
            "Apple announces new AI features",
            "Stock market reaches all-time high",
            "Championship game ends in dramatic finish"
        ]
        response = client.post(
            "/predict/batch",
            json={"texts": texts}
        )
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == len(texts)
    
    def test_predict_batch_too_many(self):
        """Test batch prediction exceeding max size."""
        texts = ["text"] * 101  # Exceeds max of 100
        response = client.post(
            "/predict/batch",
            json={"texts": texts}
        )
        assert response.status_code == 422
    
    def test_predict_batch_empty(self):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 422


class TestStatsEndpoint:
    """Test statistics endpoint."""
    
    def test_get_stats(self):
        """Test the stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "cache_hit_rate" in data
        assert "model_info" in data


class TestCacheEndpoint:
    """Test cache management endpoint."""
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        response = client.delete("/cache")
        # May return 503 if cache not available
        assert response.status_code in [200, 503]


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""
    
    def test_metrics_available(self):
        """Test that metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        # Check for expected metrics
        assert "classification_predictions_total" in content or "http_request_duration_seconds" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
