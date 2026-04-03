# Makefile for managing the development environment and Docker stack
# Usage:
#   make env         # Create a local .venv virtual environment
#   make install     # Install dependencies in the active environment
#   make test        # Run the Python test suite
#   make api         # Start the FastAPI backend
#   make ui          # Start the Chainlit frontend
#   make docker-up   # Build and start the Docker stack
#   make docker-down # Stop the Docker stack
.PHONY: help env install test api ui docker-up docker-down

# Variables for commands
PYTHON ?= python
CHAINLIT ?= chainlit
DOCKER_COMPOSE := docker compose

# Default target
help:
	@printf "Available targets:\n"
	@printf "  make env         Create a local .venv virtual environment\n"
	@printf "  make install     Install dependencies in the active environment using the requirements.txt file\n"
	@printf "  make test        Run the Python test suite\n"
	@printf "  make api         Start the FastAPI backend\n"
	@printf "  make ui          Start the Chainlit frontend\n"
	@printf "  make docker-up   Build and start the Docker stack\n"
	@printf "  make docker-down Stop the Docker stack\n"

# Targets
env:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -p '*test.py'

api:
	$(PYTHON) -m uvicorn app_fastapi:app --reload

ui:
	$(CHAINLIT) run app_chainlit.py --port 8080

docker-up:
	$(DOCKER_COMPOSE) up --build

docker-down:
	$(DOCKER_COMPOSE) down

# If you want to add more targets:
# target-name:
# 	$(command to execute)	
