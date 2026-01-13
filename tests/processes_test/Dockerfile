FROM python:3.13-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY . .

RUN pip install --no-cache-dir "numpy>=2.1.0"

CMD ["python", "tests/processes_test/processes_test.py"]
