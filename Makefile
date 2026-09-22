VENV = .venv

install:
	uv sync

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)

.PHONY: install run debug clean lint lint-strict