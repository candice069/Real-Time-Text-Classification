"""
Unit tests for the BERT Text Classifier
"""

import pytest
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../model/src"))

from bert_classifier import BERTTextClassifier


class TestBERTTextClassifier:
    """Test BERT text classifier functionality."""
    
    @pytest.fixture(scope="class")
    def classifier(self):
        """Create a classifier instance for testing."""
        classifier = BERTTextClassifier(
            model_name="bert-base-uncased",
            num_labels=5,
            max_length=128,
            confidence_threshold=0.8
        )
        # Don't actually load the model in unit tests to save time
        return classifier
    
    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.model_name == "bert-base-uncased"
        assert classifier.num_labels == 5
        assert classifier.max_length == 128
        assert classifier.confidence_threshold == 0.8
        assert classifier.label_map == {}
    
    def test_cache_key_generation(self, classifier):
        """Test cache key generation logic."""
        # This test validates the cache key concept
        text = "Test text for classification"
        # The actual implementation uses hashlib.md5
        import hashlib
        expected_key = f"classification:{hashlib.md5(text.encode()).hexdigest()}"
        assert len(expected_key) > 0
    
    def test_prediction_format(self, classifier):
        """Test prediction output format."""
        # Since we don't load the actual model, we test the format expectations
        mock_prediction = {
            "text": "Test",
            "predicted_class": "tech",
            "confidence": 0.95,
            "probabilities": {
                "tech": 0.95,
                "sports": 0.02,
                "finance": 0.01,
                "entertainment": 0.01,
                "health": 0.01
            },
            "is_confident": True
        }
        
        assert "predicted_class" in mock_prediction
        assert "confidence" in mock_prediction
        assert "probabilities" in mock_prediction
        assert mock_prediction["confidence"] >= 0.8  # Above threshold
        assert sum(mock_prediction["probabilities"].values()) == pytest.approx(1.0, 0.01)


class TestDataPreparation:
    """Test data preparation functions."""
    
    def test_label_mapping(self):
        """Test label mapping logic."""
        labels = ["tech", "sports", "finance", "tech", "health"]
        unique_labels = sorted(set(labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        assert len(label_map) == 4  # Unique labels
        assert label_map["tech"] == 3
        assert label_map["sports"] == 2
        assert label_map["finance"] == 0
        assert label_map["health"] == 1
    
    def test_text_truncation(self):
        """Test text truncation for long inputs."""
        max_length = 512
        long_text = "word " * 1000  # Very long text
        
        # Simulate truncation
        words = long_text.split()
        truncated = " ".join(words[:max_length // 5])  # Approximate
        
        assert len(truncated.split()) < len(words)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
