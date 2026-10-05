.PHONY: format format-check lint \
	test-unit test-integration test-serving test \
	check pipeline \
	bento-build bento-image \
	migrate-image migrate \
	docker-up docker-start docker-down docker-ps docker-logs docker-restart docker-clean \
	airflow-up airflow-start airflow-down airflow-ps airflow-logs airflow-restart airflow-clean \
	k8s-namespace k8s-apply k8s-delete k8s-status k8s-clean k8s-logs k8s-rollout k8s-restart \
	k8s-images k8s-services k8s-monitoring k8s-prometheus k8s-grafana k8s-api k8s-predict

# Format code.
format:
	black .

# Check formatting.
format-check:
	black --check .

# Run lint checks.
lint:
	ruff check .

# Run unit tests.
test-unit:
	pytest tests/unit -v

# Run integration tests.
test-integration:
	pytest tests/integration -v

# Run serving tests.
test-serving:
	pytest tests/serving -v

# Run all tests.
test: test-unit test-integration test-serving

# Run all checks.
check: format-check lint test

# Build the BentoML package.
bento-build:
	python -m wine_quality_prediction.pipeline.stage_07_model_packaging
	bentoml build -f bentofile.yaml

# Build the BentoML Docker image.
bento-image:
	bentoml containerize wine_quality_service:latest \
		-t wine-quality-service:latest

# Build the migration image.
migrate-image:
	docker build \
		-f deployment/docker/Dockerfile.migrate \
		-t wine-quality-migrate:latest .

# Run database migrations.
migrate:
	docker run --rm \
		--env-file .env \
		wine-quality-migrate:latest

# Start development services.
docker-up:
	docker compose -f docker-compose.dev.yml up -d --build

# Start services without rebuilding.
docker-start:
	docker compose -f docker-compose.dev.yml up -d

# Check development services.
docker-ps:
	docker compose -f docker-compose.dev.yml ps

# Follow development logs.
docker-logs:
	docker compose -f docker-compose.dev.yml logs -f

# Restart development services.
docker-restart:
	docker compose -f docker-compose.dev.yml restart

# Stop development services.
docker-down:
	docker compose -f docker-compose.dev.yml down

# Remove development resources.
docker-clean:
	docker compose -f docker-compose.dev.yml down -v --remove-orphans

# Start Airflow.
airflow-up:
	docker compose -f docker-compose.airflow.yml up -d --build

# Start Airflow without rebuilding.
airflow-start:
	docker compose -f docker-compose.airflow.yml up -d

# Check Airflow services.
airflow-ps:
	docker compose -f docker-compose.airflow.yml ps

# Follow Airflow logs.
airflow-logs:
	docker compose -f docker-compose.airflow.yml logs -f

# Restart Airflow.
airflow-restart:
	docker compose -f docker-compose.airflow.yml restart

# Stop Airflow.
airflow-down:
	docker compose -f docker-compose.airflow.yml down

# Remove Airflow resources.
airflow-clean:
	docker compose -f docker-compose.airflow.yml down -v --remove-orphans

# Create the Kubernetes namespace.
k8s-namespace:
	kubectl create namespace wine-quality --dry-run=client -o yaml | kubectl apply -f -

# Deploy the Kubernetes resources.
k8s-apply: k8s-namespace
	kubectl apply -k deployment/k8s/base
	kubectl wait \
		--for=condition=complete \
		job/wine-quality-migrate \
		-n wine-quality \
		--timeout=120s

# Check Kubernetes resources.
k8s-status:
	kubectl get pods -n wine-quality
	kubectl get services -n wine-quality
	kubectl get deployments -n wine-quality

# Clean old migration jobs.
k8s-clean:
	kubectl delete job -n wine-quality \
		-l app=wine-quality-migrate \
		--ignore-not-found

# Check the API rollout.
k8s-rollout:
	kubectl rollout status deployment/wine-quality-service -n wine-quality

# Follow API logs.
k8s-logs:
	kubectl logs -f deployment/wine-quality-service -n wine-quality

# Restart the API deployment.
k8s-restart:
	kubectl rollout restart deployment/wine-quality-service -n wine-quality

# Remove the Kubernetes resources.
k8s-delete:
	kubectl delete -k deployment/k8s/base

# Access the API locally.
k8s-api:
	kubectl port-forward svc/wine-quality-service 3000:3000 -n wine-quality

# Follow API logs.
k8s-logs:
	-kubectl logs -f deployment/wine-quality-service -n wine-quality

# Send a sample prediction.
# Send a sample prediction.
k8s-predict:
	curl -X POST http://127.0.0.1:3000/api/v1/predict \
		-H "Content-Type: application/json" \
		-d '{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11.0,"total_sulfur_dioxide":34.0,"density":0.9978,"ph":3.51,"sulphates":0.56,"alcohol":9.4}'

# Access Prometheus locally.
k8s-prometheus:
	kubectl port-forward svc/prometheus 9090:9090 -n wine-quality

# Access Grafana locally.
k8s-grafana:
	kubectl port-forward svc/grafana 3001:3000 -n wine-quality
