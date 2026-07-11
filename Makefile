.PHONY: install test dashboard clean

install:
	poetry install

test:
	poetry run pytest tests/

dashboard:
	poetry run streamlit run src/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
