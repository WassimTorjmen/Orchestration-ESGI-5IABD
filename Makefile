# ==============================================================================
# Projet de classification - Makefile (squelette)
# ==============================================================================
# Seuls les targets d'INSTALLATION sont fournis. Les autres sont a completer
# au fil des TP (un `# TODO (Sx)` indique la commande attendue).
# Environnement gere par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := uv run python
RUN          := uv run
VENV_DIR     := .venv
PYTHONPATH   ?= src
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5000
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna evaluate evaluate-no-validate mlflow api frontend \
        docker-build docker-run docker-up docker-train docker-down \
        airflow-init airflow-up airflow-down \
        lint format type test check


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup - Installation de l'environnement Python (uv + pyproject.toml) [FOURNI]
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Creation du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv cree$(RESET)"

deps-sync: check-uv ## Synchronise les dependances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra train --extra api --extra frontend --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

install: deps-sync ## Cree le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock genere$(RESET)"

reset-env: check-uv ## Reinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Reinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"


# ==============================================================================
# Pipeline ML  [A COMPLETER]
# ==============================================================================

data: ## Prepare/genere le jeu de donnees dans data/
	# TODO (S0) : appeler votre script de preparation de donnees

train: ## Entraine la baseline -> models/model.joblib (C=.. MAX_ITER=..)
	$(PYTHON) -m mlproject.train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Compare RF / XGBoost / LightGBM (GridSearchCV) + SHAP (CV=.. SCORING=..)
	$(PYTHON) -m mlproject.train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise RF / XGBoost / LightGBM avec Optuna (N_TRIALS=.. CV=..)
	$(PYTHON) -m mlproject.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

evaluate: ## Evalue le modele du registry et applique la porte qualite
	$(PYTHON) -m mlproject.evaluate

evaluate-no-validate: ## Evalue le modele du registry sans porte qualite
	$(PYTHON) -m mlproject.evaluate --no-validate

mlflow: ## Demarre le serveur MLflow sur le port 5000
	$(RUN) mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT) \
	  --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

api: ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	$(RUN) uvicorn mlproject.api:app --reload --host $(API_HOST) --port $(API_PORT)

api-health: ## Verifie que l'API repond (/health)
	curl -s http://$(API_HOST):$(API_PORT)/health | python3 -m json.tool

api-predict: ## Envoie une requete de prediction exemple
	curl -s -X POST http://$(API_HOST):$(API_PORT)/predict \
	  -H "Content-Type: application/json" \
	  -d '{"Pclass":1,"Age":29.0,"SibSp":0,"Parch":0,"Fare":211.3,"FamilySize":1,"IsAlone":1,"Sex":"female","Embarked":"S","Title":"Miss"}' \
	  | python3 -m json.tool

api-info: ## Affiche la version du modele servi (/model-info)
	curl -s http://$(API_HOST):$(API_PORT)/model-info | python3 -m json.tool

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT, API_URL)
	$(RUN) streamlit run src/frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker  [A COMPLETER]
# ==============================================================================

docker-build: ## Construit l'image d'entrainement
	docker build -f src/docker/Dockerfile.train -t mlproject-train .

docker-run: ## Lance l'entrainement en conteneur (standalone)
	docker run --rm -v "$(CURDIR)/models:/app/models" mlproject-train

docker-train: ## Lance l'entrainement via docker compose (alimente le volume)
	docker compose --profile train run --rm train

docker-up: ## Demarre la stack (mlflow, api, frontend)
	docker compose up -d --build mlflow api frontend

docker-down: ## Arrete et supprime les conteneurs (conserve les volumes)
	docker compose down


# ==============================================================================
# Airflow  [S17]
# ==============================================================================

AIRFLOW_DAGS_DIR := $(CURDIR)/src/dags

airflow-init: ## Initialise Airflow (a lancer une seule fois)
	@echo "$(YELLOW)>> Initialisation Airflow...$(RESET)"
	mkdir -p airflow/logs airflow/plugins
	echo "AIRFLOW_UID=$(shell id -u)" > airflow/.env
	echo "AIRFLOW_DAGS_FOLDER=$(AIRFLOW_DAGS_DIR)" >> airflow/.env
	curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml' \
	  --output-dir airflow/
	cd airflow && docker compose up airflow-init
	@echo "$(GREEN)[OK] Airflow initialise$(RESET)"

airflow-up: ## Demarre Airflow (UI sur http://localhost:8080)
	@echo "$(YELLOW)>> Demarrage Airflow...$(RESET)"
	cd airflow && AIRFLOW__CORE__DAGS_FOLDER=$(AIRFLOW_DAGS_DIR) docker compose up -d
	@echo "$(GREEN)[OK] Airflow demarre : http://localhost:8080 (airflow/airflow)$(RESET)"

airflow-down: ## Arrete Airflow
	cd airflow && docker compose down


# ==============================================================================
# Qualite  [A COMPLETER]
# ==============================================================================

lint: ## Verifie le style (ruff check)
	$(RUN) ruff check src/

format: ## Formate le code (ruff format)
	$(RUN) ruff format src/

type: ## Verifie les types (mypy)
	$(RUN) mypy src/mlproject

test: ## Lance les tests (pytest)
	$(RUN) pytest --cov=mlproject --cov-report=term-missing -q

check: lint type test ## Workflow qualite complet (lint + types + tests)
