FROM python:3.13-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY . .

RUN pip install --no-cache-dir "numpy>=2.1.0"

CMD ["python", "tests/sequential_test/sequential_test.py"]

# old
# FROM debian:bookworm-slim

# # ---- system ----
# RUN apt-get update && apt-get install -y \
#     wget \
#     build-essential \
#     libssl-dev \
#     zlib1g-dev \
#     libncurses5-dev \
#     libnss3-dev \
#     libreadline-dev \
#     libffi-dev \
#     libsqlite3-dev \
#     libbz2-dev \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # ---- Python 3.13.1 ----
# WORKDIR /usr/src
# RUN wget https://www.python.org/ftp/python/3.13.1/Python-3.13.1.tgz \
#     && tar -xf Python-3.13.1.tgz

# WORKDIR /usr/src/Python-3.13.1
# RUN ./configure --enable-optimizations \
#     && make -j$(nproc) \
#     && make altinstall

# # python -> python3.13 (UWAGA: bez "t")
# RUN ln -s /usr/local/bin/python3.13 /usr/local/bin/python

# # ---- pip ----
# RUN python -m ensurepip \
#     && python -m pip install --upgrade pip

# # ---- app ----
# WORKDIR /app
# ENV PYTHONPATH=/app
# COPY . .

# RUN python -m pip install "numpy>=2.1.0"

# CMD ["python", "tests/sequential_test/sequential_test.py"]
