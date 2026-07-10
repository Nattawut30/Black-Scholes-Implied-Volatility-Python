.PHONY: install install-dev test type-check check run clean

# Install runtime dependencies only.
install:
	python3.11 -m pip install -r requirements.txt

# Install everything, including dev tools (pytest, mypy).
install-dev:
	python3.11 -m pip install -r requirements.txt
	python3.11 -m pip install pytest mypy

# Run the test suite.
test:
	PYTHONPATH=. python3.11 -m pytest tests/

# Static type checking. Catches type errors before runtime.
type-check:
	python3.11 -m mypy src/ --ignore-missing-imports --disable-error-code attr-defined

# One command to type-check and test. Run this before pushing.
check: type-check test

# Launch the Streamlit dashboard locally on http://localhost:8501
run:
	python3.11 -m streamlit run src/streamlit_app.py

# Clean up Python and tool caches.
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
