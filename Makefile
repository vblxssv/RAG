VENV = .venv


install:
	uv sync


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)



.PHONY: install run debug clean lint lint-strict
