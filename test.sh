#!/bin/bash

# Quick Test Runner for CraftNudge
# This script provides quick access to common testing commands

set -e

echo "🧪 CraftNudge Test Runner"
echo "=========================="

case "$1" in
    "quick")
        echo "Running quick tests..."
        python -m pytest tests/ -v --tb=short
        ;;
    "coverage")
        echo "Running tests with coverage..."
        python -m pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=100
        ;;
    "full")
        echo "Running full test suite..."
        python run_tests.py
        ;;
    "lint")
        echo "Running linting..."
        python -m flake8 . --max-line-length=100 --ignore=E203,W503
        python -m black . --check
        ;;
    "format")
        echo "Formatting code..."
        python -m black .
        python -m isort .
        ;;
    "type")
        echo "Running type checking..."
        python -m mypy . --ignore-missing-imports
        ;;
    "unit")
        echo "Running unit tests..."
        python -m pytest tests/ -m unit -v
        ;;
    "integration")
        echo "Running integration tests..."
        python -m pytest tests/ -m integration -v
        ;;
    "help"|"-h"|"--help")
        echo "Usage: ./test.sh [command]"
        echo ""
        echo "Commands:"
        echo "  quick       - Run quick tests without coverage"
        echo "  coverage    - Run tests with coverage report"
        echo "  full        - Run full test suite with all checks"
        echo "  lint        - Run code linting"
        echo "  format      - Format code with black and isort"
        echo "  type        - Run type checking"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  help        - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './test.sh help' for usage information"
        exit 1
        ;;
esac
