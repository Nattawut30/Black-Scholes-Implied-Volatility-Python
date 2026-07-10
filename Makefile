.PHONY: install-dev check test

install-dev:
	poetry install

check:
	poetry run mypy src/ --ignore-missing-imports --disable-error-code attr-defined

test:
	PYTHONPATH=. poetry run pytest tests/
