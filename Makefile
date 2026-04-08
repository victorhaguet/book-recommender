# Makefile for managing the development environment and Docker stack
# Usage:
#   make env         # Create a local .venv virtual environment
#   make install     # Install the project and core dependencies
#   make install-hf  # Install optional Hugging Face embedding dependencies
#   make test        # Run the Python test suite
#   make coverage    # Run the Python test suite with coverage enforcement
#   make coverage-html Generate an HTML coverage report in htmlcov/
#   make coverage-file FILE=src/path.py Show missing coverage lines for one file
#   make api         # Start the FastAPI backend
#   make ui          # Start the Chainlit frontend
#   make docker-up   # Build and start the Docker stack
#   make docker-down # Stop the Docker stack
.PHONY: help env install install-hf test coverage coverage-html coverage-file api ui docker-up docker-down

# Variables for commands
PYTHON ?= python
CHAINLIT ?= chainlit
DOCKER_COMPOSE := docker compose

# Default target
help:
	@printf "Available targets:\n"
	@printf "  make env         Create a local .venv virtual environment\n"
	@printf "  make install     Install the project and core dependencies in editable mode\n"
	@printf "  make install-hf  Install optional Hugging Face embedding dependencies\n"
	@printf "  make test        Run the Python test suite\n"
	@printf "  make coverage    Run the Python test suite with coverage enforcement\n"
	@printf "  make coverage-html Generate an HTML coverage report in htmlcov/\n"
	@printf "  make coverage-file FILE=src/path.py Show missing coverage lines for one file\n"
	@printf "  make api         Start the FastAPI backend\n"
	@printf "  make ui          Start the Chainlit frontend\n"
	@printf "  make docker-up   Build and start the Docker stack\n"
	@printf "  make docker-down Stop the Docker stack\n"

# Targets
env:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-hf:
	$(PYTHON) -m pip install -e ".[hf]"

test:
	$(PYTHON) -m unittest discover -s tests -p '*test.py'

coverage:
	$(PYTHON) -m coverage run -m unittest discover -s tests -p '*test.py'
	$(PYTHON) -m coverage report

coverage-html:
	$(PYTHON) -m coverage run -m unittest discover -s tests -p '*test.py'
	$(PYTHON) -m coverage html

coverage-file:
	@test -n "$(FILE)" || (printf "Usage: make coverage-file FILE=src/path/to/file.py\n" >&2; exit 1)
	$(PYTHON) -m coverage run -m unittest discover -s tests -p '*test.py'
	$(PYTHON) -m coverage report -m $(FILE)

api:
	$(PYTHON) -m uvicorn src.app_fastapi:app --reload

ui:
	PYTHONPATH=. $(CHAINLIT) run src/app_chainlit.py --port 8080

docker-up:
	$(DOCKER_COMPOSE) up --build

docker-down:
	$(DOCKER_COMPOSE) down

# If you want to add more targets:
# target-name:
# 	$(command to execute)	
