# Makefile for Low-Rank SHAP project
# Provides convenient commands for development, testing, and reproducibility

.PHONY: help install install-dev test test-coverage lint format clean reproduce paper package release

# Default target
help:
	@echo "Low-Rank SHAP Project Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run test suite"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo ""
	@echo "Reproducibility:"
	@echo "  reproduce    Reproduce all experiments from scratch"
	@echo "  paper        Generate research paper PDF"
	@echo ""
	@echo "Packaging:"
	@echo "  package      Build distribution packages"
	@echo "  release      Create release (tag + upload)"
	@echo "  clean        Clean build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=lowrank_shap --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 lowrank_shap/ tests/ scripts/
	black --check lowrank_shap/ tests/ scripts/

format:
	black lowrank_shap/ tests/ scripts/

# Reproducibility
reproduce:
	@echo "Reproducing Low-Rank SHAP experiments..."
	@echo "Step 1: Running validation tests..."
	python scripts/test_week3.py
	@echo "Step 2: Running debug tests..."
	python scripts/debug_week3.py
	@echo "Step 3: Running full experiments..."
	python scripts/week3_experiments.py
	@echo "Step 4: Analyzing results..."
	python scripts/analyze_week3_results.py
	@echo "✅ All experiments completed successfully!"

# Paper generation
paper:
	@echo "Generating research paper..."
	cd paper && quarto render paper.qmd --to pdf
	@echo "✅ Paper generated: paper/paper.pdf"

# Package management
package:
	python -m build

release: clean test package
	@echo "Creating release..."
	git tag -a v1.0.0 -m "Release version 1.0.0"
	git push origin v1.0.0
	python -m twine upload dist/*

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

quick-test:
	python -c "import lowrank_shap; print('✅ Package imports successfully')"
	python -c "from lowrank_shap import LowRankSHAP; print('✅ Core classes available')"

# Experiment shortcuts
run-experiments:
	python scripts/week3_experiments.py

analyze-results:
	python scripts/analyze_week3_results.py

debug-pipeline:
	python scripts/debug_week3.py
