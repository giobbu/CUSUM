# Makefile

.PHONY: help install backend-local-build backend-local-down

help:
	@echo "MLOps Cronjob Commands:"
	@echo "  make install               - Install dependencies (uv sync)"
	@echo "  make backend-local-build   - Build and run backend locally"
	@echo "  make backend-local-down    - Remove backend image"

install:
	uv sync

backend-local-build:
	docker build -f dockerfile.backend -t detection-cronjob-backend .
	docker run -d -v "$$PWD":/home -it detection-cronjob-backend

backend-local-down:
	docker rmi $(docker ps -a -q --filter ancestor=detection-cronjob-backend) -f || true

