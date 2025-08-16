.PHONY: help install dev-install test lint format clean build docs serve-docs docker-build docker-run

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := calute
DOCKER_IMAGE := $(PROJECT_NAME):latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install the package
	$(PIP) install --upgrade pip
	$(PIP) install -e .

dev-install: ## Install the package with development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,tools,llm-providers,database,monitoring]"
	pre-commit install

test: ## Run tests with coverage
	pytest tests/ -v --cov=$(PROJECT_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml

test-fast: ## Run tests without coverage
	pytest tests/ -v

test-watch: ## Run tests in watch mode
	pytest-watch tests/ -- -v

lint: ## Run linting checks
	@echo "$(YELLOW)Running ruff...$(NC)"
	ruff check $(PROJECT_NAME)/
	@echo "$(YELLOW)Running black check...$(NC)"
	black --check $(PROJECT_NAME)/
	@echo "$(YELLOW)Running mypy...$(NC)"
	mypy $(PROJECT_NAME)/ --ignore-missing-imports
	@echo "$(YELLOW)Running bandit...$(NC)"
	bandit -r $(PROJECT_NAME)/ -f screen
	@echo "$(GREEN)All linting checks passed!$(NC)"

format: ## Format code
	@echo "$(YELLOW)Running isort...$(NC)"
	isort $(PROJECT_NAME)/ tests/
	@echo "$(YELLOW)Running black...$(NC)"
	black $(PROJECT_NAME)/ tests/
	@echo "$(YELLOW)Running ruff fix...$(NC)"
	ruff check --fix $(PROJECT_NAME)/ tests/
	@echo "$(GREEN)Code formatted successfully!$(NC)"

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Cleaned build artifacts!$(NC)"

build: clean ## Build distribution packages
	$(PYTHON) -m build
	twine check dist/*
	@echo "$(GREEN)Build complete! Packages in dist/$(NC)"

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

docs: ## Build documentation
	cd docs && make clean && make html
	@echo "$(GREEN)Documentation built! Open docs/_build/html/index.html$(NC)"

serve-docs: docs ## Build and serve documentation
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE)$(NC)"

docker-run: ## Run Docker container
	docker run -it --rm $(DOCKER_IMAGE)

docker-push: ## Push Docker image to registry
	docker tag $(DOCKER_IMAGE) erfanzar/$(PROJECT_NAME):latest
	docker push erfanzar/$(PROJECT_NAME):latest

# Development commands
dev-server: ## Run development server
	$(PYTHON) -m $(PROJECT_NAME).server --debug

dev-shell: ## Start interactive Python shell with project imported
	$(PYTHON) -c "from $(PROJECT_NAME) import *; import IPython; IPython.embed()"

# Database commands
db-migrate: ## Run database migrations
	alembic upgrade head

db-rollback: ## Rollback database migration
	alembic downgrade -1

db-reset: ## Reset database
	alembic downgrade base
	alembic upgrade head

# Quality checks
check-security: ## Run security checks
	safety check
	pip-audit

check-deps: ## Check for outdated dependencies
	pip list --outdated

update-deps: ## Update dependencies
	pip install --upgrade -r requirements.txt

# Git hooks
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Release commands
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

changelog: ## Generate changelog
	git-changelog -o CHANGELOG.md

release: test lint ## Create a new release
	@echo "$(YELLOW)Creating new release...$(NC)"
	@read -p "Version type (patch/minor/major): " version_type; \
	bump2version $$version_type; \
	git push && git push --tags
	@echo "$(GREEN)Release created!$(NC)"

# Benchmarking
benchmark: ## Run performance benchmarks
	$(PYTHON) -m pytest tests/benchmarks/ -v --benchmark-only

profile: ## Profile the application
	$(PYTHON) -m cProfile -o profile.stats run.py
	$(PYTHON) -m pstats profile.stats

# Utility commands
count-lines: ## Count lines of code
	@echo "$(YELLOW)Lines of code:$(NC)"
	@find $(PROJECT_NAME) -name "*.py" -type f -exec wc -l {} + | sort -rn

todo: ## Show all TODO comments
	@echo "$(YELLOW)TODO items:$(NC)"
	@grep -r "TODO\|FIXME\|XXX" $(PROJECT_NAME)/ --color=always || echo "No TODOs found!"

contributors: ## Show contributors
	@echo "$(YELLOW)Contributors:$(NC)"
	@git shortlog -sn

.DEFAULT_GOAL := help