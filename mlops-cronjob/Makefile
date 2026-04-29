# Makefile

.PHONY: help sync docker backend-up backend-down frontend-up frontend-down

help:
	@echo "MLOps Cronjob Commands:"
	@echo "  make sync - Install dependencies (uv sync)"
	@echo "  make docker - start docker daemon"
	@echo "  make backend-up - Build backend image and run container"
	@echo "  make frontend-up - Build frontend image and run container"
	@echo "  make backend-down - Remove backend image"
	@echo "  make frontend-down - Stop and remove frontend container and image"

sync:
	uv sync

docker:
	open -a Docker

backend-up:
	docker build -f dockerfile.backend -t detection-cronjob-backend .
	docker run -d -v "$$PWD":/home -it detection-cronjob-backend

frontend-up:
	docker build -f dockerfile.frontend -t streamlit-frontend .
	docker run -p 8501:8501 -v "$$PWD":/home -it streamlit-frontend

backend-down:
	docker rmi $(docker ps -a -q --filter ancestor=detection-cronjob-backend) -f || true

frontend-down:
	docker rm -f $(shell docker ps -a -q --filter ancestor=streamlit-frontend)
	docker rmi -f $(shell docker images -q streamlit-frontend)

