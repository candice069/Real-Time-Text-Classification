# 实时文本分类系统 - 中文讲解

## 一、项目概述

### 1.1 这是一个什么项目？

这是一个**实时文本分类系统**，可以自动识别文本属于哪个类别（比如：科技、体育、金融、娱乐、健康等）。

### 1.2 举个例子理解

想象你运营一个新闻网站，每天收到成千上万篇文章。你需要：
- 自动把文章分到不同栏目
- 处理速度要快（每秒1万+篇）
- 分类准确率需使用独立数据集验证

这个系统就是解决这个问题的！

---

## 二、系统架构图解

```
用户/文章源
    │
    ▼
┌─────────────┐
│  发送请求    │  把文章发送到系统
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Kafka     │  消息队列（高速缓冲通道）
│  (消息队列)  │  可以处理每秒1万+条消息
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  API服务器   │  FastAPI提供HTTP接口
│  (FastAPI)  │  接收单个或批量请求
└──────┬──────┘
       │
    ┌──┴──┐
    │     │
    ▼     ▼
┌──────┐ ┌──────┐
│Redis │ │模型   │  Redis检查缓存
│缓存  │ │推理   │  BERT模型进行分类
└──┬───┘ └──────┘
   │
   ▼
┌─────────────┐
│  返回结果    │  告诉你是哪类+可信度
└─────────────┘
```

---

## 三、核心组件详解

### 3.1 BERT 模型（"大脑"）

**位置**：`model/src/bert_classifier.py`

#### 什么是BERT？
BERT是Google开发的一种**人工智能语言理解模型**，特点：
- 能"理解"上下文（不只是看单个词）
- 在大量文本上预训练过
- 可以微调做特定任务（比如分类）

#### 代码中的关键部分：

```python
class BERTTextClassifier:
    def __init__(self):
        self.model_name = "bert-base-uncased"  # 使用BERT基础版
        self.max_length = 512                   # 最多处理512个字
        self.confidence_threshold = 0.8        # 可信度门槛80%
```

#### 工作流程：

1. **训练阶段**（train.py）
   ```
   输入：大量已标注的文章（如"这篇文章是科技类"）
   ↓
   BERT学习这些文章的"特征"
   ↓
   输出：训练好的模型文件
   ```

2. **预测阶段**
   ```
   输入：新文章
   ↓
   BERT分析文章
   ↓
   输出：
   - 预测类别（如"科技"）
   - 可信度（如95%）
   - 各类别概率分布
   ```

---

### 3.2 Kafka 流处理（"高速公路"）

**位置**：`streaming/producer/kafka_producer.py` 和 `streaming/consumer/kafka_consumer.py`

#### 为什么需要Kafka？

想象春运时的火车站：
- **没有Kafka**：每个乘客直接挤上车，混乱不堪
- **有Kafka**：排队候车，有序上车，效率高

Kafka就是一个**消息队列系统**，让数据有序流动。

#### 生产者（Producer）- 发送端

```python
# 简化理解
class ClassificationKafkaProducer:
    def send_batch(self, requests):
        # 把1000条消息打包
        # 一次性发送给Kafka
        # 发送速率目标示例，非实测吞吐量
```

**关键配置**：
- `batch_size=10000`：一次打包1万条
- `compression_type="lz4"`：压缩消息，省带宽
- `linger.ms=10`：等10毫秒攒一批再发

#### 消费者（Consumer）- 接收端

```python
class ClassificationKafkaConsumer:
    def run(self):
        # 持续监听Kafka
        # 收到消息就处理
        # 4个线程并行处理
```

**为什么用多线程？**
- 单线程：1个人干活
- 4线程：4个人同时干活，速度快4倍

---

### 3.3 Redis 缓存（"记忆助手"）

**位置**：`api/src/main.py` 中的缓存逻辑

#### 为什么需要缓存？

场景：同一篇文章被请求了1000次

**没有缓存**：
```
每次都要BERT推理 → 耗时500ms × 1000次 = 500秒
```

**有缓存**：
```
第1次：BERT推理 → 500ms → 结果存入Redis
第2-1000次：直接从Redis取 → 5ms × 999次 = 5秒
```

**节省了99%的时间！**

#### 缓存实现：

```python
async def predict(request):
    # 1. 生成缓存键（基于文章内容）
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    # 2. 先查Redis
    cached = await redis.get(cache_key)
    if cached:
        return cached  # 命中缓存，直接返回
    
    # 3. 缓存没有，调用BERT推理
    result = model.predict(text)
    
    # 4. 结果存入Redis（1小时过期）
    await redis.setex(cache_key, 3600, result)
    
    return result
```

---

### 3.4 FastAPI 后端（"服务台"）

**位置**：`api/src/main.py`

#### 什么是FastAPI？

FastAPI是一个现代化的Python Web框架，特点：
- 速度快（比Flask快很多）
- 自动生成API文档
- 类型检查，减少bug

#### 提供的API接口：

| 接口 | 功能 | 示例 |
|------|------|------|
| `POST /predict` | 单条分类 | 发一篇文章，返回类别 |
| `POST /predict/batch` | 批量分类 | 一次发100篇 |
| `GET /health` | 健康检查 | 看服务是否正常 |
| `GET /stats` | 统计信息 | 缓存命中率等 |
| `DELETE /cache` | 清空缓存 | 管理员用 |

#### 代码示例：

```python
@app.post("/predict")
async def predict(request: ClassificationRequest):
    # 1. 检查缓存
    # 2. 如果缓存没有，调用模型
    # 3. 记录Prometheus指标
    # 4. 返回结果
```

---

### 3.5 Prometheus + Grafana（"监控大屏"）

**位置**：`monitoring/` 目录

#### 监控什么？

```
┌─────────────────────────────────────┐
│           Grafana 仪表盘            │
├─────────────────────────────────────┤
│  处理速度: 5,000 篇/秒              │
│  平均延迟: 50ms                     │
│  缓存命中率: 65%                    │
│  错误率: 0.1%                       │
│                                     │
│  [CPU使用率折线图]                  │
│  [内存使用折线图]                   │
│  [请求延迟分布热力图]               │
└─────────────────────────────────────┘
```

#### 关键指标：

1. **classification_predictions_total** - 总请求数
2. **classification_prediction_duration_seconds** - 推理耗时
3. **classification_cache_hits_total** - 缓存命中次数
4. **classification_active_requests** - 正在处理的请求数

#### 告警规则（alerts）：

```yaml
# 延迟太高就告警
- alert: ClassificationHighLatency
  expr: latency > 1.0  # 超过1秒
  
# 错误率太高就告警  
- alert: ClassificationHighErrorRate
  expr: error_rate > 0.05  # 超过5%
```

---

### 3.6 Kubernetes（"自动化运维"）

**位置**：`k8s/base/` 目录

#### 什么是Kubernetes？

Kubernetes（简称K8s）是**容器编排平台**，自动管理你的服务：

**传统部署**：
```
服务器挂了 → 人工发现 → 登录服务器 → 重启服务
（耗时30分钟）
```

**Kubernetes部署**：
```
服务挂了 → 自动检测 → 自动重启
（耗时30秒）
```

#### 自动扩缩容（HPA）：

```yaml
spec:
  minReplicas: 2      # 最少2个实例
  maxReplicas: 10     # 最多10个实例
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70  # CPU超70%就扩容
```

**场景**：
- 凌晨3点流量小 → 2个实例运行
- 早上9点流量大 → 自动扩容到8个实例
- 晚上10点流量降 → 自动缩容回2个

**省钱又保证服务！**

---

## 四、数据流转完整示例

让我们跟踪一篇文章的分类过程：

```
1. 用户上传文章
   "苹果发布新款AI芯片，性能提升50%"
   
2. 发送到Kafka生产者
   → 打包成消息格式
   → 加入消息队列
   
3. Kafka消费者取出消息
   → 转发给API服务器
   
4. API服务器处理
   → 生成缓存键："a1b2c3d4..."
   → 查Redis：缓存未命中
   → 调用BERT模型推理
   
5. BERT模型分析
   → 输入："苹果发布新款AI芯片..."
   → 特征提取：提到"苹果""AI""芯片"
   → 输出：科技类，95%可信度
   
6. 返回并缓存
   → 结果存入Redis（1小时有效）
   → 记录Prometheus指标
   → 返回给用户：{"category": "科技", "confidence": 0.95}
   
7. 5分钟后，同一篇文章又来了
   → 查Redis：缓存命中！
   → 直接返回，只用了5ms（省了495ms）
```

---

## 五、文件结构说明

```
real-time-text-classification/
│
├── api/src/main.py              ← 网站后端（FastAPI）
│   - 提供HTTP接口
│   - 处理缓存（Redis）
│   - 收集指标（Prometheus）
│
├── model/src/
│   ├── bert_classifier.py      ← AI大脑（BERT模型）
│   └── train.py                ← 模型训练脚本
│
├── streaming/
│   ├── producer/               ← 高速发送器（Kafka生产者）
│   └── consumer/               ← 高速接收器（Kafka消费者）
│
├── docker/                     ← Docker镜像配置
│   - 定义如何打包成容器
│
├── k8s/base/                   ← Kubernetes配置
│   - 定义如何在集群中运行
│
├── monitoring/                  ← 监控配置
│   ├── prometheus/             - 数据收集
│   └── grafana/                - 可视化大屏
│
├── docker-compose.yml          ← 本地一键启动
├── Makefile                    ← 常用命令快捷方式
└── README.md                   ← 英文文档
```

---

## 六、常用操作指南

### 6.1 本地开发（最简单）

```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 启动基础服务（Redis、Kafka、监控）
docker-compose up -d redis kafka prometheus grafana

# 3. 训练模型（第一次需要）
cd model/src
python train.py

# 4. 启动API服务
cd api/src
uvicorn main:app --reload

# 5. 测试
打开浏览器访问：http://localhost:8000/docs
```

### 6.2 Docker Compose 启动（推荐）

```bash
# 一键启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止所有服务
docker-compose down
```

### 6.3 Kubernetes 部署（生产环境）

```bash
# 构建镜像
make docker-build

# 部署到K8s
make k8s-deploy

# 查看状态
make k8s-status

# 查看日志
make k8s-logs
```

---

## 七、常见问题

### Q1: 为什么用这么多技术？

每个技术解决特定问题：
- **BERT**：解决"理解文本内容"问题
- **Kafka**：解决"海量数据传输"问题
- **Redis**：解决"重复计算"问题
- **FastAPI**：解决"对外提供服务"问题
- **K8s**：解决"自动运维"问题
- **Prometheus**：解决"了解系统状态"问题

### Q2: 可以只学其中一部分吗？

可以！建议学习顺序：
1. 先看 `api/src/main.py`（最容易理解）
2. 再看 `model/src/bert_classifier.py`（理解AI部分）
3. 然后看 `streaming/`（理解流处理）
4. 最后看 `k8s/`（运维部分）

### Q3: 如何修改分类类别？

编辑 `model/src/train.py`，修改类别定义：

```python
CATEGORIES = {
    "科技": [...],
    "体育": [...],
    "金融": [...],
    # 添加你的新类别
    "教育": [...],
    "旅游": [...],
}
```

然后重新训练模型。

---

## 八、学习资源推荐

1. **BERT理解**：
   - 搜索"BERT模型图解"
   - 了解"预训练+微调"概念

2. **Kafka学习**：
   - 搜索"Kafka消息队列入门"
   - 理解"生产者-消费者"模式

3. **Redis学习**：
   - 搜索"Redis缓存原理"
   - 了解"内存数据库"概念

4. **FastAPI学习**：
   - 官方文档：https://fastapi.tiangolo.com/zh/
   - 有完整中文文档

5. **Kubernetes学习**：
   - 搜索"Kubernetes入门"
   - 从"Pod、Service、Deployment"基本概念开始

---

## 九、总结

这个项目展示了如何构建一个**企业级的实时文本分类系统**：

1. **BERT** 提供高质量的AI分类能力
2. **Kafka** 保证海量数据的可靠传输
3. **Redis** 大幅提升响应速度
4. **FastAPI** 提供简洁高效的API服务
5. **K8s** 实现自动化运维和弹性扩缩容
6. **Prometheus/Grafana** 让系统状态一目了然

这是一个完整的**微服务架构**示例，各个组件松耦合，可以独立开发、部署、扩展。

---

**如有疑问，欢迎随时提问！**
