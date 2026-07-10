FROM python:3.10-slim
WORKDIR /app
RUN pip install poetry
COPY . /app
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
EXPOSE 7860
CMD ["poetry", "run", "streamlit", "run", "src/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
