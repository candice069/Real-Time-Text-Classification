"""
Training script for BERT-based text classification model.
"""

import os
import argparse
import json
import logging
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from bert_classifier import BERTTextClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_data() -> Tuple[List[str], List[int], List[str]]:
    """
    Generate sample training data for demonstration.
    In production, replace with actual data loading logic.
    """
    # Sample data for multi-class classification (categories: tech, sports, finance, entertainment, health)
    texts = []
    labels = []
    
    # Tech category (0)
    tech_samples = [
        "Apple announces new iPhone with revolutionary AI capabilities",
        "Google Cloud launches new machine learning platform for enterprises",
        "Microsoft unveils quantum computing breakthrough at tech conference",
        "Tesla's autopilot system receives major safety update",
        "OpenAI releases GPT-5 with enhanced reasoning capabilities",
        "NVIDIA announces next-generation GPU architecture",
        "Amazon Web Services introduces new serverless computing features",
        "Meta reveals plans for immersive virtual reality workspace",
        "Samsung's new foldable phone sets sales records",
        "Intel announces breakthrough in chip manufacturing process",
        "SpaceX successfully launches next-generation satellite constellation",
        "GitHub introduces AI-powered code review assistant",
        "Cloud computing adoption accelerates across industries",
        "Blockchain technology transforms supply chain management",
        "5G networks enable new possibilities for IoT devices",
    ] * 100
    
    # Sports category (1)
    sports_samples = [
        "Championship final goes into overtime with dramatic finish",
        "Olympic gold medalist breaks world record in 100m sprint",
        "NBA playoffs feature intense rivalry matchup",
        "Premier League title race goes down to final matchday",
        "Grand Slam champion announces retirement after 20-year career",
        "Underdog team defeats defending champions in upset victory",
        "Marathon runner completes epic comeback after injury",
        "Formula 1 introduces new regulations for next season",
        "Tennis star wins fourth consecutive major tournament",
        "NFL draft sees record number of quarterbacks selected",
        "Soccer legend transfers to rival club in shock move",
        "Golf tournament ends with playoff hole drama",
        "Esports championship draws record viewership numbers",
        "Swimmer sets new world best time in butterfly event",
        "Boxing match ends in controversial decision",
    ] * 100
    
    # Finance category (2)
    finance_samples = [
        "Federal Reserve signals potential interest rate changes",
        "Stock market reaches all-time high amid economic optimism",
        "Cryptocurrency market volatility continues with major swings",
        "Global supply chain disruptions impact quarterly earnings",
        "Tech IPO raises record funding in public debut",
        "Hedge fund reports exceptional returns for investors",
        "Real estate market shows signs of cooling",
        "Central bank announces quantitative easing measures",
        "Corporate merger creates industry-leading conglomerate",
        "Inflation data exceeds analyst expectations",
        "Venture capital funding reaches historic levels",
        "Oil prices fluctuate amid geopolitical tensions",
        "Banking sector reports strong quarterly profits",
        "Currency exchange rates impact export competitiveness",
        "Pension fund announces sustainable investment strategy",
    ] * 100
    
    # Entertainment category (3)
    entertainment_samples = [
        "Blockbuster movie breaks box office records worldwide",
        "Streaming service releases hit original series",
        "Music festival announces star-studded lineup",
        "Celebrity couple announces engagement on social media",
        "Award-winning director announces ambitious new project",
        "Broadway show receives critical acclaim",
        "Streaming wars intensify with exclusive content deals",
        "Celebrity memoir becomes instant bestseller",
        "Film festival showcases international cinema",
        "Pop star announces world tour with record ticket sales",
        "Reality show finale draws massive audience",
        "Video game adaptation exceeds fan expectations",
        "Comedy special breaks viewing records",
        "Documentary reveals untold story of historic event",
        "Orchestra performs sold-out concert series",
    ] * 100
    
    # Health category (4)
    health_samples = [
        "Medical researchers announce breakthrough in cancer treatment",
        "New vaccine shows promising results in clinical trials",
        "Healthcare system adopts AI for diagnostic assistance",
        "Mental health awareness campaign reaches millions",
        "Study reveals benefits of Mediterranean diet",
        "Fitness trends shift toward holistic wellness approaches",
        "Telemedicine adoption accelerates post-pandemic",
        "Gene therapy shows promise for rare diseases",
        "Hospital network implements cutting-edge surgical robots",
        "Pharmaceutical company receives FDA approval",
        "Global health initiative targets infectious diseases",
        "Wearable technology advances personal health monitoring",
        "Nutrition research challenges conventional wisdom",
        "Yoga and meditation gain mainstream acceptance",
        "Public health campaign targets smoking cessation",
    ] * 100
    
    texts.extend(tech_samples + sports_samples + finance_samples + 
                 entertainment_samples + health_samples)
    labels.extend([0] * len(tech_samples) +
                  [1] * len(sports_samples) +
                  [2] * len(finance_samples) +
                  [3] * len(entertainment_samples) +
                  [4] * len(health_samples))
    
    label_names = ["tech", "sports", "finance", "entertainment", "health"]
    
    return texts, labels, label_names


def main():
    parser = argparse.ArgumentParser(description="Train BERT text classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (JSON format with 'text' and 'label' fields)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/bert-classifier",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained BERT model to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Load data
    if args.data_path:
        logger.info(f"Loading data from {args.data_path}")
        with open(args.data_path, "r") as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        labels = [item["label"] for item in data]
        label_names = None
    else:
        logger.info("Using sample training data")
        texts, labels, label_names = load_sample_data()
    
    # Log dataset statistics
    logger.info(f"Total samples: {len(texts)}")
    logger.info(f"Unique labels: {len(set(labels))}")
    for label in sorted(set(labels)):
        count = labels.count(label)
        label_name = label_names[label] if label_names else label
        logger.info(f"  {label_name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Initialize classifier
    classifier = BERTTextClassifier(
        model_name=args.model_name,
        num_labels=len(set(labels)),
        max_length=args.max_length
    )
    classifier.initialize()
    
    # Train model
    results = classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Log final results
    logger.info("Training completed successfully!")
    logger.info(f"Final validation F1 (weighted): {results['metrics']['f1_weighted']:.4f}")
    logger.info(f"Final validation accuracy: {results['metrics']['accuracy']:.4f}")
    
    # Test inference speed
    test_texts = val_texts[:100]
    import time
    start_time = time.time()
    predictions = classifier.predict(test_texts, batch_size=32)
    inference_time = time.time() - start_time
    
    logger.info(f"\nInference Performance:")
    logger.info(f"  Samples processed: {len(test_texts)}")
    logger.info(f"  Total time: {inference_time:.2f}s")
    logger.info(f"  Throughput: {len(test_texts)/inference_time:.1f} docs/sec")
    logger.info(f"  Latency: {inference_time/len(test_texts)*1000:.1f}ms/doc")


if __name__ == "__main__":
    main()
