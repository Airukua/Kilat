.PHONY: help install install-dev lint lint-fix format format-check type-check check test verify chaos clean build pre-commit pre-commit-install

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PIP ?= $(PYTHON) -m pip
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
PYTEST ?= $(PYTHON) -m pytest
PRE_COMMIT ?= $(PYTHON) -m pre_commit

help: ## Show this help message
	@echo "Kilat Development Makefile"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package in production mode
	$(PIP) install .

install-dev: ## Install package in editable mode with development dependencies
	$(PIP) install -e ".[dev]"

lint: ## Run ruff linter on source code
	$(RUFF) check src

lint-fix: ## Run ruff linter with automatic fixes
	$(RUFF) check --fix src

format: ## Format code with ruff formatter
	$(RUFF) format src

format-check: ## Check code formatting without making changes
	$(RUFF) format --check src

type-check: ## Run mypy type checker on source code
	$(MYPY) src

check: lint type-check ## Run standard quality checks (lint and type-check)

test: ## Run test suite if tests directory exists
	@if [ -d "tests" ]; then \
		$(PYTEST) tests; \
	else \
		echo "No tests/ directory found. Skipping unit tests."; \
	fi

verify: ## Run 6-Gate Pre-Training Assurance verification on debug_cpu config
	$(PYTHON) -m kilat.cli verify --config configs/debug_cpu.yaml --output-dir verify_signoff

chaos: ## Run Active ML Chaos Engineering Saboteur suite (C-01..C-08)
	$(PYTHON) -m kilat.cli chaos --scenarios all

certify: ## Run Kilat Model Architecture Certification Pipeline (KMCP)
	$(PYTHON) -m kilat.cli certify --models kilat-macro-synergy --pillar all --output-dir verify_signoff

pre-commit: ## Run pre-commit hooks on all files
	$(PRE_COMMIT) run --all-files

pre-commit-install: ## Install pre-commit hooks into git repository
	$(PRE_COMMIT) install

clean: ## Clean build, distribution, bytecode, and cache files
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/ .eggs/ artifacts/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/

build: ## Build source distribution and wheel packages
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build
