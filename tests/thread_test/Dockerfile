FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libnss3-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src
RUN wget https://www.python.org/ftp/python/3.13.1/Python-3.13.1.tgz \
    && tar -xf Python-3.13.1.tgz

WORKDIR /usr/src/Python-3.13.1
RUN ./configure --disable-gil --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall

RUN ln -s /usr/local/bin/python3.13t /usr/local/bin/python

RUN python -c "import sysconfig; print(f'GIL DISABLED: {sysconfig.get_config_var('Py_GIL_DISABLED')}')"

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip && \
    python -m pip install numpy>=2.1.0

CMD ["python", "tests/thread_test/thread_test.py"]