.PHONY: install dev api ui test lint fmt docker samples ollama-setup

install:
	pip install -e ".[dev]"

api:
	uvicorn app.api.main:app --reload --port 8000

ui:
	streamlit run app/ui/streamlit_app.py

test:
	pytest

lint:
	ruff check app tests

fmt:
	ruff format app tests
	ruff check --fix app tests

samples:
	python scripts/make_samples.py

ollama-setup:
	python scripts/setup_ollama.py

docker:
	docker compose up --build
