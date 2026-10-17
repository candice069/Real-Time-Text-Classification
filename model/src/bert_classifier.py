"""
BERT-based Text Classification Model
BERT-based multi-class classification; evaluation results are not yet verified
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import tensorflow as tf
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
    TFTrainer,
    TFTrainingArguments,
    create_optimizer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTTextClassifier:
    """
    BERT-based multi-class text classifier optimized for real-time inference.
    
    Features:
    - Fine-tuned BERT-base model
    - Multi-class classification support
    - Optimized for high-throughput inference
    - Includes evaluation methods; model quality requires measurement
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 5,
        max_length: int = 512,
        confidence_threshold: float = 0.8
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tokenizer = None
        self.label_map = {}
        self.reverse_label_map = {}
        
    def initialize(self, model_path: Optional[str] = None):
        """Initialize tokenizer and model."""
        logger.info(f"Initializing BERT classifier with {self.model_name}")
        
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name,
            cache_dir="/tmp/bert_cache"
        )
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = TFBertForSequenceClassification.from_pretrained(model_path)
            # Load label mapping
            label_map_path = os.path.join(model_path, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path, "r") as f:
                    self.label_map = json.load(f)
                    self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        else:
            logger.info("Loading pre-trained BERT model")
            self.model = TFBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                cache_dir="/tmp/bert_cache"
            )
            
    def prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 32
    ) -> tf.data.Dataset:
        """Prepare and tokenize data for training."""
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="tf"
        )
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels
        ))
        
        return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def compute_metrics(self, labels, predictions) -> Dict[str, float]:
        """Compute classification metrics including F1 score."""
        preds = np.argmax(predictions, axis=1)
        
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_micro": f1_score(labels, preds, average="micro"),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }
        
        return metrics
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        output_dir: str = "./models/bert-classifier"
    ) -> Dict:
        """
        Fine-tune BERT model on domain-specific data.
        
        Training Configuration:
        - Optimized for multi-class classification
        - Learning rate warmup for stable training
        - Weight decay for regularization
        """
        logger.info(f"Starting training with {len(train_texts)} samples")
        logger.info(f"Validation samples: {len(val_texts)}")
        
        # Create label mapping if not exists
        unique_labels = sorted(set(train_labels + val_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Remap labels to continuous indices
        train_labels_mapped = [self.label_map[l] for l in train_labels]
        val_labels_mapped = [self.label_map[l] for l in val_labels]
        
        # Prepare datasets
        train_dataset = self.prepare_data(train_texts, train_labels_mapped, batch_size)
        val_dataset = self.prepare_data(val_texts, val_labels_mapped, batch_size)
        
        # Calculate training steps
        num_train_steps = (len(train_texts) // batch_size) * epochs
        
        # Create optimizer with warmup
        optimizer, schedule = create_optimizer(
            init_lr=learning_rate,
            num_warmup_steps=warmup_steps,
            num_train_steps=num_train_steps,
            weight_decay_rate=weight_decay,
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, "checkpoint-{epoch:02d}"),
                save_best_only=True,
                monitor="val_f1_weighted"
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on validation set
        val_predictions = self.model.predict(val_dataset)
        metrics = self.compute_metrics(
            np.array(val_labels_mapped),
            val_predictions.logits
        )
        
        logger.info("Validation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save model and tokenizer
        self.save(output_dir)
        
        return {
            "history": history.history,
            "metrics": metrics
        }
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Perform batch prediction on input texts.
        
        Returns:
            List of prediction results with class, confidence, and probabilities
        """
        if not isinstance(texts, list):
            texts = [texts]
            
        # Tokenize inputs
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="tf"
        )
        
        # Run inference
        outputs = self.model.predict(dict(encodings))
        logits = outputs.logits
        
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        # Format results
        results = []
        for i, (pred, conf, probs) in enumerate(zip(predictions, confidences, probabilities)):
            predicted_label = self.reverse_label_map.get(int(pred), int(pred))
            
            result = {
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "predicted_class": predicted_label,
                "confidence": float(conf),
                "probabilities": {
                    self.reverse_label_map.get(j, j): float(prob)
                    for j, prob in enumerate(probs)
                },
                "is_confident": conf >= self.confidence_threshold
            }
            results.append(result)
            
        return results
    
    def predict_single(self, text: str) -> Dict:
        """Predict a single text input."""
        return self.predict([text])[0]
    
    def save(self, output_dir: str):
        """Save model, tokenizer, and label mapping."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        with open(os.path.join(output_dir, "label_map.json"), "w") as f:
            json.dump(self.label_map, f, indent=2)
            
        # Save model config
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "confidence_threshold": self.confidence_threshold
        }
        with open(os.path.join(output_dir, "classifier_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info("Model saved successfully")
    
    @classmethod
    def load(cls, model_path: str):
        """Load a trained classifier from disk."""
        # Load config
        config_path = os.path.join(model_path, "classifier_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            classifier = cls(
                model_name=config.get("model_name", "bert-base-uncased"),
                num_labels=config.get("num_labels", 5),
                max_length=config.get("max_length", 512),
                confidence_threshold=config.get("confidence_threshold", 0.8)
            )
        else:
            classifier = cls()
            
        classifier.initialize(model_path)
        return classifier
