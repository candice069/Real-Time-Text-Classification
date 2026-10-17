.PHONY: help install train test docker-up docker-down k8s-deploy k8s-delete clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install Python dependencies"
	@echo "  train         - Train the BERT classifier model"
	@echo "  test          - Run tests"
	@echo "  docker-up     - Start all services with Docker Compose"
	@echo "  docker-down   - Stop Docker Compose services"
	@echo "  k8s-deploy    - Deploy to Kubernetes"
	@echo "  k8s-delete    - Delete Kubernetes deployment"
	@echo "  clean         - Clean up generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Train the model
train:
	cd model/src && python train.py --epochs 3 --batch-size 16

# Run tests
test:
	pytest tests/ -v

# Docker commands
docker-up:
	docker-compose up -d redis kafka prometheus grafana

docker-up-all:
	docker-compose --profile producer --profile consumer up -d

docker-down:
	docker-compose down -v

docker-logs:
	docker-compose logs -f api

# Build Docker images
docker-build:
	docker build -t text-classification-api:latest -f docker/Dockerfile.api .
	docker build -t text-classification-producer:latest -f docker/Dockerfile.producer .
	docker build -t text-classification-consumer:latest -f docker/Dockerfile.consumer .

# Kubernetes commands
k8s-deploy:
	kubectl apply -k k8s/base/

k8s-delete:
	kubectl delete -k k8s/base/

k8s-status:
	kubectl get pods -n text-classification
	kubectl get svc -n text-classification
	kubectl get hpa -n text-classification

k8s-logs:
	kubectl logs -f deployment/classification-api -n text-classification

k8s-port-forward:
	kubectl port-forward svc/classification-api-service 8000:80 -n text-classification &
	kubectl port-forward svc/prometheus 9090:9090 -n text-classification &
	kubectl port-forward svc/grafana 3000:3000 -n text-classification &

# Monitoring
monitor-open:
	@echo "Opening monitoring dashboards..."
	open http://localhost:9090  # Prometheus
	open http://localhost:3000  # Grafana
	open http://localhost:8080  # Kafka UI

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -delete
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# API testing
api-test:
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "Apple announces new AI-powered iPhone features"}'

api-health:
	curl http://localhost:8000/health

api-stats:
	curl http://localhost:8000/stats

# Performance benchmark
benchmark:
	@echo "Running throughput benchmark..."
	docker-compose --profile producer run --rm producer python -m streaming.producer.kafka_producer --benchmark --target-throughput 10000 --duration 60
