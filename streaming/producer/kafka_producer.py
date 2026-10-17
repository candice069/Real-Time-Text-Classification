"""
Kafka Producer for High-Throughput Text Classification Requests
Uses batching and compression; throughput requires measurement
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import random

from confluent_kafka import Producer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationRequest:
    """Data class for classification request."""
    request_id: str
    text: str
    timestamp: str
    source: str
    metadata: Optional[Dict] = None


class ClassificationKafkaProducer:
    """
    High-performance Kafka producer for text classification requests.
    
    Features:
    - Batch processing with configurable throughput targets
    - Message compression for efficient network usage
    - Asynchronous delivery with callbacks
    - Automatic retry and error handling
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "text-classification-requests",
        batch_size: int = 10000,
        linger_ms: int = 10,
        compression_type: str = "lz4"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.batch_size = batch_size
        
        # Configure producer for high throughput
        self.config = {
            "bootstrap.servers": bootstrap_servers,
            "batch.num.messages": batch_size,
            "linger.ms": linger_ms,
            "compression.type": compression_type,
            "acks": "1",  # Balance between durability and performance
            "retries": 3,
            "retry.backoff.ms": 100,
            "max.in.flight.requests.per.connection": 5,
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.kbytes": 1048576,  # 1GB
            "queue.buffering.max.ms": 100,
        }
        
        self.producer = None
        self.delivery_reports = []
        self.messages_sent = 0
        self.messages_failed = 0
        
    def initialize(self):
        """Initialize Kafka producer and ensure topic exists."""
        logger.info(f"Initializing Kafka producer for {self.bootstrap_servers}")
        
        # Create producer
        self.producer = Producer(self.config)
        
        # Ensure topic exists
        self._ensure_topic()
        
        logger.info("Kafka producer initialized successfully")
    
    def _ensure_topic(self, num_partitions: int = 6, replication_factor: int = 1):
        """Create topic if it doesn't exist."""
        admin_client = AdminClient({
            "bootstrap.servers": self.bootstrap_servers
        })
        
        topic_metadata = admin_client.list_topics(timeout=5)
        
        if self.topic not in topic_metadata.topics:
            logger.info(f"Creating topic {self.topic} with {num_partitions} partitions")
            new_topic = NewTopic(
                topic=self.topic,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
                config={
                    "retention.ms": "604800000",  # 7 days
                    "cleanup.policy": "delete"
                }
            )
            
            fs = admin_client.create_topics([new_topic])
            
            for topic, future in fs.items():
                try:
                    future.result()
                    logger.info(f"Topic {topic} created successfully")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info(f"Topic {topic} already exists")
                    else:
                        logger.error(f"Failed to create topic {topic}: {e}")
    
    def _delivery_callback(self, err, msg):
        """Handle message delivery callback."""
        if err:
            self.messages_failed += 1
            logger.error(f"Message delivery failed: {err}")
        else:
            self.messages_sent += 1
            if self.messages_sent % 10000 == 0:
                logger.info(f"Messages sent: {self.messages_sent}")
    
    def send_request(
        self,
        request: ClassificationRequest,
        key: Optional[str] = None
    ):
        """Send a single classification request."""
        message = {
            "request_id": request.request_id,
            "text": request.text,
            "timestamp": request.timestamp,
            "source": request.source,
            "metadata": request.metadata or {}
        }
        
        # Use request_id as key for partitioning
        message_key = key or request.request_id
        
        self.producer.produce(
            topic=self.topic,
            key=message_key.encode("utf-8"),
            value=json.dumps(message).encode("utf-8"),
            callback=self._delivery_callback
        )
    
    def send_batch(
        self,
        requests: List[ClassificationRequest],
        flush_interval: int = 1000
    ) -> Dict[str, int]:
        """
        Send a batch of classification requests efficiently.
        
        Returns:
            Dictionary with send statistics
        """
        start_time = datetime.now()
        
        for i, request in enumerate(requests):
            self.send_request(request)
            
            # Trigger delivery every N messages for backpressure control
            if i % flush_interval == 0:
                self.producer.poll(0)
        
        # Wait for all messages to be delivered
        logger.info(f"Flushing {len(requests)} messages to Kafka...")
        self.producer.flush()
        
        duration = (datetime.now() - start_time).total_seconds()
        throughput = len(requests) / duration if duration > 0 else 0
        
        stats = {
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "duration_seconds": duration,
            "throughput": throughput
        }
        
        logger.info(f"Batch sent: {stats['messages_sent']} messages in {duration:.2f}s "
                   f"({throughput:.1f} msgs/sec)")
        
        return stats
    
    def send_texts(
        self,
        texts: List[str],
        source: str = "api",
        metadata: Optional[Dict] = None
    ) -> Dict[str, int]:
        """
        Convenience method to send raw texts for classification.
        
        Args:
            texts: List of text strings to classify
            source: Source identifier for the request
            metadata: Optional metadata to include
            
        Returns:
            Dictionary with send statistics
        """
        import uuid
        
        requests = [
            ClassificationRequest(
                request_id=str(uuid.uuid4()),
                text=text,
                timestamp=datetime.utcnow().isoformat(),
                source=source,
                metadata=metadata
            )
            for text in texts
        ]
        
        return self.send_batch(requests)
    
    def close(self):
        """Close the producer connection."""
        if self.producer:
            logger.info("Flushing remaining messages...")
            self.producer.flush()
            logger.info("Producer closed")
    
    def get_stats(self) -> Dict:
        """Get producer statistics."""
        return {
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "success_rate": (
                self.messages_sent / (self.messages_sent + self.messages_failed)
                if (self.messages_sent + self.messages_failed) > 0 else 0
            )
        }


class SimulatedDataProducer:
    """
    Simulated data generator for testing and benchmarking.
    Generates realistic text classification requests at scale.
    """
    
    CATEGORIES = {
        "tech": [
            "New AI breakthrough enables real-time language translation",
            "Cloud computing costs decrease with new optimization techniques",
            "5G networks expand coverage to rural areas",
            "Cybersecurity threats increase with remote work adoption",
            "Quantum computing research accelerates",
        ],
        "sports": [
            "Championship game ends in dramatic overtime victory",
            "Star athlete breaks long-standing world record",
            "New training methods improve athlete performance",
            "International tournament draws record viewership",
            "Youth sports programs receive major funding boost",
        ],
        "finance": [
            "Stock markets react to economic policy changes",
            "Cryptocurrency regulations impact trading volumes",
            "Interest rate decisions affect mortgage applications",
            "Tech IPO generates significant investor interest",
            "Global trade agreements boost market confidence",
        ],
        "entertainment": [
            "Blockbuster film breaks opening weekend records",
            "Streaming service releases exclusive content",
            "Music festival lineup features diverse artists",
            "Award show celebrates industry achievements",
            "Virtual concerts reach global audiences",
        ],
        "health": [
            "Medical research reveals new treatment approach",
            "Public health campaign improves vaccination rates",
            "Wearable devices track fitness metrics accurately",
            "Mental health awareness increases support programs",
            "Nutrition study challenges dietary recommendations",
        ]
    }
    
    def __init__(self, producer: ClassificationKafkaProducer):
        self.producer = producer
    
    def generate_request(
        self,
        category: Optional[str] = None
    ) -> ClassificationRequest:
        """Generate a single classification request."""
        import uuid
        
        if category is None:
            category = random.choice(list(self.CATEGORIES.keys()))
        
        template = random.choice(self.CATEGORIES[category])
        
        # Add some variation
        variations = [
            f"Breaking: {template}",
            f"Report: {template}",
            f"Analysis: {template}",
            f"Update: {template}",
            f"News: {template}"
        ]
        
        text = random.choice(variations)
        
        return ClassificationRequest(
            request_id=str(uuid.uuid4()),
            text=text,
            timestamp=datetime.utcnow().isoformat(),
            source="simulator",
            metadata={"category": category, "is_simulated": True}
        )
    
    def run_benchmark(
        self,
        target_throughput: int = 10000,
        duration_seconds: int = 60
    ) -> Dict:
        """
        Run a throughput benchmark.
        
        Args:
            target_throughput: Target messages per second
            duration_seconds: How long to run the benchmark
            
        Returns:
            Benchmark statistics
        """
        import time
        
        logger.info(f"Starting benchmark: {target_throughput} msgs/sec for {duration_seconds}s")
        
        start_time = time.time()
        messages_sent = 0
        batch_size = min(1000, target_throughput)
        
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            # Generate batch
            requests = [
                self.generate_request()
                for _ in range(batch_size)
            ]
            
            # Send batch
            self.producer.send_batch(requests)
            messages_sent += batch_size
            
            # Calculate sleep time to maintain target throughput
            elapsed = time.time() - batch_start
            target_time = batch_size / target_throughput
            sleep_time = max(0, target_time - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        total_time = time.time() - start_time
        actual_throughput = messages_sent / total_time
        
        stats = {
            "messages_sent": messages_sent,
            "duration_seconds": total_time,
            "target_throughput": target_throughput,
            "actual_throughput": actual_throughput,
            "efficiency": actual_throughput / target_throughput * 100
        }
        
        logger.info(f"Benchmark completed: {stats}")
        return stats


def main():
    """Main entry point for producer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kafka Producer for text classification")
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC", "text-classification-requests"),
        help="Kafka topic name"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run throughput benchmark"
    )
    parser.add_argument(
        "--target-throughput",
        type=int,
        default=10000,
        help="Target throughput for benchmark"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds"
    )
    
    args = parser.parse_args()
    
    # Initialize producer
    producer = ClassificationKafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic
    )
    producer.initialize()
    
    try:
        if args.benchmark:
            # Run benchmark
            simulator = SimulatedDataProducer(producer)
            simulator.run_benchmark(
                target_throughput=args.target_throughput,
                duration_seconds=args.duration
            )
        else:
            # Send sample data
            simulator = SimulatedDataProducer(producer)
            requests = [simulator.generate_request() for _ in range(1000)]
            producer.send_batch(requests)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
