.PHONY: install test demo lint serve docker clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install with all extras (api + training + dev)
	python -m venv .venv
	.venv/bin/pip install -e ".[api,training,dev]"

test: ## Run all 42 tests
	.venv/bin/python -m pytest tests/ -v --tb=short

demo: ## Run the villain adaptive intelligence demo
	.venv/bin/python examples/villain_demo.py

demo-village: ## Run the village social simulation demo
	.venv/bin/python examples/village_demo.py

demo-lod: ## Run the LOD scaling demo (50 agents)
	.venv/bin/python examples/lod_demo.py

demo-combat: ## Run the combat simulation demo
	.venv/bin/python examples/run_simulation.py

serve: ## Start the FastAPI inference server on port 8787
	.venv/bin/uvicorn opennpc.api.service:app --host 0.0.0.0 --port 8787 --reload

docker: ## Build and run the Docker container
	docker build -t opennpc-api .
	docker run -p 8787:8787 opennpc-api

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.sqlite3" -delete 2>/dev/null || true
