FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app

COPY . .

RUN pip install --no-cache-dir \
    numpy>=2.1.0 \
    mpi4py

CMD ["mpirun", "--allow-run-as-root", "-np", "2", "python", "tests/mpi_test/mpi_test.py"]

