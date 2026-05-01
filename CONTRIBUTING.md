# Contributing to OpenNPC

Thanks for your interest in contributing to OpenNPC! This guide covers the development workflow.

## Setup

```bash
git clone https://github.com/your-org/openNPC.git
cd openNPC
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,training,dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests must pass before submitting a PR.

## Project Structure

```
opennpc/           Core SDK modules
opennpc/simulation/  Simulation environments
opennpc/training/    RL training pipelines
opennpc/api/         FastAPI inference service
adapters/            Unity/Unreal/Godot engine adapters
configs/             Agent configuration templates
examples/            Runnable demos
tests/               Unit tests
docs/                Architecture & walkthrough
```

## Adding a New Module

1. Create the module in the appropriate package.
2. Export public API from `opennpc/__init__.py`.
3. Add tests in `tests/test_<module>.py`.
4. Add a runnable demo in `examples/` if the module is user-facing.
5. Update `docs/architecture.md` with the new layer/component.

## Adding a New Agent Config

1. Create a JSON file in `configs/` following the existing templates.
2. Required fields: `name`, `role`, `goals` (with priorities), `personality`, `constraints`, `valid_actions`.
3. Test by loading it in `examples/basic_decision.py`.

## Code Standards

- **Type hints** on all public functions.
- **Docstrings** on all classes and public methods.
- **Max function length:** 40 lines. If longer, extract.
- **Max file length:** 300 lines. If longer, split by responsibility.
- **No `any` types** — use `Unknown` with type guards.
- **Tests pass before commit** — `pytest tests/ -v` must be green.

## Submitting Changes

1. Fork the repo and create a feature branch.
2. Make your changes with tests.
3. Run `python -m pytest tests/ -v` — all must pass.
4. Open a PR with a description of what changed and why.
