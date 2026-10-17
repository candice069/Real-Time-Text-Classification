# Real-Time Text Classification

A portfolio prototype combining a TensorFlow BERT classifier, a FastAPI prediction service, Redis caching, and Kafka producer/consumer components.

## Components

- `model/src/`: BERT classifier, training, saving/loading, and evaluation methods.
- `api/src/`: synchronous and asynchronous API variants with prediction, health, and metrics endpoints.
- `streaming/`: Kafka producers and consumers for classification requests.
- `docker/` and `docker-compose.yml`: local service and container configuration.
- `k8s/base/`: deployment, service, and autoscaling manifests.
- `monitoring/`: Prometheus alerts and Grafana dashboard configuration.
- `tests/`: existing test sources.

## Validation status

All 16 Python source files passed syntax parsing, and Docker Compose configuration validation passed. Full dependency installation, unit tests, trained-model inference, and end-to-end streaming have not been verified in this publication review.

No throughput, latency reduction, F1, or accuracy results are claimed. The training script defaults to repeated synthetic examples; its random train/validation split can contain duplicate texts and is unsuitable for reporting generalization quality. Supply an independently split dataset for meaningful evaluation.

The API initializes a base model when no saved classifier exists. Predictions from an untrained classification head must not be treated as evaluated classification results. The supplied Compose API service does not mount a trained-model directory; configure that separately before using a trained checkpoint.

## Development starting points

Use Python 3.10 and the pinned dependencies in `requirements.txt`. Dependency compatibility on your platform still needs validation.

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

Train the demonstration model from the repository root:

```bash
cd model/src
python train.py --epochs 3 --batch-size 16
```

Review `.env.example`, copy it to a local `.env`, and set a unique `GRAFANA_ADMIN_PASSWORD` before starting the infrastructure:

```bash
cp .env.example .env
# Edit .env and set GRAFANA_ADMIN_PASSWORD.
docker compose config --quiet
docker compose up -d redis kafka prometheus grafana
```

Compose host ports are bound to localhost for local development. Review configuration, model paths, and service connectivity before starting the API and streaming profiles. Kubernetes custom metrics require a separately configured metrics adapter; the manifests alone do not provide one.

## Monitoring

Local dashboard ports: Prometheus `9090`, Grafana `3000`, and Kafka UI `8080`. Grafana uses username `admin` and the password you configure locally.

## Reproducibility

Benchmark and evaluate with documented hardware, dataset provenance and split, dependency versions, exact commands, and saved outputs before adding performance claims.
