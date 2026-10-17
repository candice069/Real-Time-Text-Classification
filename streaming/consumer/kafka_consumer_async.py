"""
异步模式Consumer - 直接处理消息，不再调API
适合：API转Kafka的架构模式
"""

import os
import sys
import json
import logging
import time
from typing import Optional, Dict
import threading
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "../../model/src"))
from bert_classifier import BERTTextClassifier

from confluent_kafka import Consumer, KafkaError
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncProcessor:
    """
    异步处理器 - Consumer直接处理消息
    
    流程：
    1. 从Kafka拉取消息
    2. 加载BERT模型进行推理
    3. 结果写入Redis
    4. 如果有callback_url，发送回调通知
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "text-classification-requests",
        group_id: str = "async-processor",
        model_path: str = "./models/bert-classifier"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.group_id = group_id
        self.model_path = model_path
        
        # 初始化组件
        self.consumer = None
        self.redis_client = None
        self.model: Optional[BERTTextClassifier] = None
        
        # 统计
        self.processed_count = 0
        self.failed_count = 0
        
    def init(self):
        """初始化所有组件"""
        logger.info("Initializing async processor...")
        
        # 1. 初始化Kafka Consumer
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "max.poll.interval.ms": 300000,
        }
        self.consumer = Consumer(config)
        self.consumer.subscribe([self.input_topic])
        
        # 2. 初始化Redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # 3. 加载BERT模型
        logger.info(f"Loading model from {self.model_path}")
        if os.path.exists(self.model_path):
            self.model = BERTTextClassifier.load(self.model_path)
        else:
            logger.warning("Model not found, initializing new model")
            self.model = BERTTextClassifier()
            self.model.initialize()
        
        logger.info("Initialization complete")
    
    def process_message(self, msg) -> bool:
        """
        处理单条消息
        
        与之前的Consumer不同，这里直接调用模型，不再调API
        """
        try:
            data = json.loads(msg.value().decode("utf-8"))
            request_id = data.get("request_id")
            text = data.get("text")
            callback_url = data.get("callback_url")
            
            logger.info(f"Processing request: {request_id}")
            
            # 1. 更新Redis状态为"处理中"
            self._update_task_status(request_id, "processing")
            
            # 2. 直接调用BERT模型推理
            start_time = time.time()
            prediction = self.model.predict_single(text)
            processing_time = (time.time() - start_time) * 1000
            
            # 3. 构建完整结果
            result = {
                "request_id": request_id,
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "processing_time_ms": processing_time,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            
            # 4. 结果存入Redis
            self._store_result(request_id, result)
            
            # 5. 更新任务状态为"已完成"
            self._update_task_status(request_id, "completed", result=result)
            
            # 6. 发送回调通知（如果有callback_url）
            if callback_url:
                self._send_callback(callback_url, result)
            
            self.processed_count += 1
            
            if self.processed_count % 100 == 0:
                logger.info(f"Processed {self.processed_count} messages")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            self.failed_count += 1
            # 可以在这里实现重试逻辑或发送到死信队列
            return False
    
    def _update_task_status(self, request_id: str, status: str, result: Optional[Dict] = None):
        """更新任务状态到Redis"""
        task_data = self.redis_client.get(f"task:{request_id}")
        if task_data:
            task = json.loads(task_data)
            task["status"] = status
            if result:
                task["result_summary"] = {
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"]
                }
            self.redis_client.setex(
                f"task:{request_id}",
                3600,
                json.dumps(task)
            )
    
    def _store_result(self, request_id: str, result: Dict):
        """存储详细结果"""
        self.redis_client.setex(
            f"result:{request_id}",
            3600,  # 1小时过期
            json.dumps(result)
        )
    
    def _send_callback(self, callback_url: str, result: Dict):
        """发送回调通知"""
        try:
            response = requests.post(
                callback_url,
                json=result,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                logger.info(f"Callback sent successfully to {callback_url}")
            else:
                logger.warning(f"Callback failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send callback: {e}")
    
    def run(self):
        """主循环"""
        logger.info("Starting async processor...")
        
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaError(msg.error())
                
                # 处理消息
                self.process_message(msg)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.consumer.close()
    
    def get_stats(self):
        """获取处理统计"""
        return {
            "processed": self.processed_count,
            "failed": self.failed_count,
            "success_rate": self.processed_count / (self.processed_count + self.failed_count)
            if (self.processed_count + self.failed_count) > 0 else 0
        }


def main():
    processor = AsyncProcessor()
    processor.init()
    processor.run()


if __name__ == "__main__":
    main()
