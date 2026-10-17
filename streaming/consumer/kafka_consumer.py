"""
Kafka Consumer for Real-Time Text Classification
Processes streaming requests and sends results to output topic
"""

import os
import json
import logging
import signal
import sys
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import time

from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Data class for classification result."""
    request_id: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    is_confident: bool


class ClassificationKafkaConsumer:
    """
    High-performance Kafka consumer for real-time text classification.
    
    Features:
    - Multi-threaded message processing
    - Automatic offset management
    - Dead letter queue for failed messages
    - Backpressure handling
    - Graceful shutdown
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "text-classification-requests",
        output_topic: str = "text-classification-results",
        dlq_topic: str = "text-classification-dlq",
        group_id: str = "classifier-group",
        api_endpoint: str = "http://localhost:8000/predict",
        max_poll_records: int = 500,
        worker_threads: int = 4,
        max_retries: int = 3
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.dlq_topic = dlq_topic
        self.group_id = group_id
        self.api_endpoint = api_endpoint
        self.max_poll_records = max_poll_records
        self.worker_threads = worker_threads
        self.max_retries = max_retries
        
        # Consumer configuration optimized for throughput
        self.consumer_config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 45000,
            "heartbeat.interval.ms": 15000,
            "max.partition.fetch.bytes": 1048576 * 10,  # 10MB
            "fetch.max.bytes": 1048576 * 50,  # 50MB
            "fetch.min.bytes": 1,
            "fetch.wait.max.ms": 100,
        }
        
        # Producer config for output topic
        self.producer_config = {
            "bootstrap.servers": bootstrap_servers,
            "batch.num.messages": 1000,
            "linger.ms": 10,
            "compression.type": "lz4",
            "acks": "1",
            "retries": 3,
        }
        
        self.consumer = None
        self.producer = None
        self.running = False
        
        # Metrics
        self.messages_processed = 0
        self.messages_failed = 0
        self.processing_times = []
        self.start_time = None
        
        # Threading
        self.worker_pool = []
        self.message_queue = []
        self.queue_lock = threading.Lock()
        
    def initialize(self):
        """Initialize Kafka consumer and producer."""
        logger.info(f"Initializing consumer for group {self.group_id}")
        
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([self.input_topic])
        
        self.producer = Producer(self.producer_config)
        
        logger.info("Kafka consumer and producer initialized")
    
    def _send_result(
        self,
        request_id: str,
        result: Dict,
        original_key: Optional[bytes] = None
    ):
        """Send classification result to output topic."""
        message = {
            "request_id": request_id,
            "result": result,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        self.producer.produce(
            topic=self.output_topic,
            key=original_key,
            value=json.dumps(message).encode("utf-8")
        )
    
    def _send_to_dlq(
        self,
        request_id: str,
        original_message: Dict,
        error: str
    ):
        """Send failed message to dead letter queue."""
        dlq_message = {
            "request_id": request_id,
            "original_message": original_message,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
            "retry_count": original_message.get("retry_count", 0)
        }
        
        self.producer.produce(
            topic=self.dlq_topic,
            value=json.dumps(dlq_message).encode("utf-8")
        )
        
        logger.warning(f"Message {request_id} sent to DLQ: {error}")
    
    def _classify_text(self, text: str) -> Optional[Dict]:
        """
        Call the classification API to classify text.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result or None if failed
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={"text": text},
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    processing_time = (time.time() - start_time) * 1000
                    result["processing_time_ms"] = processing_time
                    return result
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Classification error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        
        return None
    
    def _process_message(self, msg) -> bool:
        """Process a single Kafka message."""
        try:
            # Parse message
            data = json.loads(msg.value().decode("utf-8"))
            request_id = data.get("request_id", "unknown")
            text = data.get("text", "")
            
            if not text:
                logger.warning(f"Empty text in message {request_id}")
                return True
            
            # Classify
            result = self._classify_text(text)
            
            if result:
                # Send result
                self._send_result(request_id, result, msg.key())
                
                # Update metrics
                self.messages_processed += 1
                if "processing_time_ms" in result:
                    self.processing_times.append(result["processing_time_ms"])
                
                # Log progress
                if self.messages_processed % 1000 == 0:
                    self._log_metrics()
                
                return True
            else:
                # Classification failed
                self.messages_failed += 1
                retry_count = data.get("retry_count", 0)
                
                if retry_count < 3:
                    data["retry_count"] = retry_count + 1
                    # Re-queue for retry (in production, use delay queue)
                    return True
                else:
                    self._send_to_dlq(request_id, data, "Max retries exceeded")
                    return True
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.messages_failed += 1
            return False
    
    def _log_metrics(self):
        """Log current processing metrics."""
        elapsed = time.time() - self.start_time if self.start_time else 1
        throughput = self.messages_processed / elapsed if elapsed > 0 else 0
        
        avg_latency = 0
        if self.processing_times:
            recent_times = self.processing_times[-1000:]
            avg_latency = sum(recent_times) / len(recent_times)
        
        logger.info(
            f"Metrics - Processed: {self.messages_processed}, "
            f"Failed: {self.messages_failed}, "
            f"Throughput: {throughput:.1f} msg/s, "
            f"Avg Latency: {avg_latency:.1f}ms"
        )
    
    def _worker_thread(self, thread_id: int):
        """Worker thread for parallel processing."""
        logger.info(f"Worker thread {thread_id} started")
        
        while self.running:
            messages = []
            
            # Get batch of messages
            with self.queue_lock:
                if self.message_queue:
                    messages = self.message_queue[:100]
                    self.message_queue = self.message_queue[100:]
            
            if not messages:
                time.sleep(0.01)
                continue
            
            # Process batch
            for msg in messages:
                try:
                    self._process_message(msg)
                except Exception as e:
                    logger.error(f"Worker {thread_id} error: {e}")
    
    def run(self):
        """Main consumer loop."""
        logger.info("Starting consumer...")
        
        self.running = True
        self.start_time = time.time()
        
        # Start worker threads
        for i in range(self.worker_threads):
            thread = threading.Thread(target=self._worker_thread, args=(i,))
            thread.daemon = True
            thread.start()
            self.worker_pool.append(thread)
        
        try:
            while self.running:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info(f"Reached end of partition {msg.topic()}")
                    else:
                        raise KafkaException(msg.error())
                else:
                    # Add to queue
                    with self.queue_lock:
                        self.message_queue.append(msg)
                    
                    # Manage backpressure
                    while len(self.message_queue) > 10000:
                        logger.warning("Backpressure: queue full, slowing down...")
                        time.sleep(0.1)
                
                # Flush producer periodically
                self.producer.poll(0)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down consumer...")
        
        self.running = False
        
        # Wait for workers
        for thread in self.worker_pool:
            thread.join(timeout=5)
        
        # Flush producer
        self.producer.flush()
        
        # Close consumer
        self.consumer.close()
        
        # Final metrics
        self._log_metrics()
        
        logger.info("Consumer shutdown complete")
    
    def get_stats(self) -> Dict:
        """Get consumer statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 1
        
        stats = {
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "elapsed_seconds": elapsed,
            "throughput": self.messages_processed / elapsed if elapsed > 0 else 0,
            "success_rate": (
                self.messages_processed / (self.messages_processed + self.messages_failed)
                if (self.messages_processed + self.messages_failed) > 0 else 0
            ),
            "avg_processing_time_ms": (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0
            )
        }
        
        return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kafka Consumer for text classification")
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--input-topic",
        default=os.getenv("KAFKA_TOPIC", "text-classification-requests"),
        help="Input topic for classification requests"
    )
    parser.add_argument(
        "--output-topic",
        default="text-classification-results",
        help="Output topic for classification results"
    )
    parser.add_argument(
        "--group-id",
        default=os.getenv("KAFKA_CONSUMER_GROUP", "classifier-group"),
        help="Consumer group ID"
    )
    parser.add_argument(
        "--api-endpoint",
        default=os.getenv("API_ENDPOINT", "http://localhost:8000/predict"),
        help="Classification API endpoint"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads"
    )
    
    args = parser.parse_args()
    
    consumer = ClassificationKafkaConsumer(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        group_id=args.group_id,
        api_endpoint=args.api_endpoint,
        worker_threads=args.workers
    )
    
    consumer.initialize()
    
    try:
        consumer.run()
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        consumer.shutdown()


if __name__ == "__main__":
    main()
