# Makefile

.PHONY: help install upgrade test

help:
	@echo "Getting Started Commands:"
	@echo "  make sync - Install dependencies (uv sync)"
	@echo "  make upgrade PACKAGE=<package-name> - Upgrade a specific package using the upgrade_package.sh script"
	@echo "  make test - Run tests with coverage using pytest"

install:
	@echo "Installing dependencies for the project..."
	uv sync

upgrade:
	@echo "Upgrading package: $(PACKAGE)"
	./upgrade_package.sh $(PACKAGE)

test:
	@echo "Running tests..."
	pytest --cov=source test/