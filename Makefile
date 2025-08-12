# === Variables ===
PYTHON              := poetry run python
PYTEST              := poetry run pytest
DOCKER_IMAGE        := rag-app
CHROMA_DB_DIR       := $(PWD)/chroma_db
CHROMA_DB_DIR_DOCKER := /app/chroma_db

# === Targets ===
.PHONY: help install test test-unit test-integration \
        lint format clean docker-build docker-run-save docker-run-retrieve

# === General Help ===
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# === Installation & Setup ===
install: ## Install dependencies using Poetry
	poetry install $(if $(PROD),--no-dev,)

# === Testing ===
test-all: ## Run all tests with pytest
	$(PYTEST) -vs

test-unit: ## Run unit tests only
	$(PYTEST) -m unit -vs

test-integration: ## Run integration tests only
	$(PYTEST) -m integration -vs

save:
	poetry run python -m src.rag.save_vector

retrieve:
	poetry run python -m src.rag.retrieve_vector


# === Linting & Formatting ===
lint: ## Run linters
	poetry run black --check .
	poetry run isort --profile black --check .

format: ## Auto-format code using Black and isort
	poetry run isort --profile black .
	poetry run black .

# === Cleaning ===
clean: ## Clean up Python cache and test artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

# === Docker Commands ===
docker-build: ## Build unified Docker image
	docker build -t $(DOCKER_IMAGE) .

docker-run-save: ## Run save operation
	docker run --rm -v "$(CHROMA_DB_DIR):$(CHROMA_DB_DIR_DOCKER)" $(DOCKER_IMAGE) save

docker-run-retrieve: ## Run retrieve operation
	docker run --rm -v "$(HOME)/.config/gcloud:/root/.config/gcloud:ro" -v "$(CHROMA_DB_DIR):$(CHROMA_DB_DIR_DOCKER)" -e GCP_PROJECT="$(GCP_PROJECT)" $(DOCKER_IMAGE) retrieve