# === Variables ===
PYTHON              := poetry run python
PYTEST              := poetry run pytest

# === Targets ===
.PHONY: help install test test-unit test-integration \
        lint format clean dev ci

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


