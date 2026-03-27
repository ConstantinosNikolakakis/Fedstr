# Stage 1: Builder
# Start FROM python:3.11 so PyO3 links against
# exactly the same libpython as the runtime stage.
# Then install Rust toolchain on top.
FROM python:3.11-slim-bookworm AS builder

# Install build tools + Rust dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Tell PyO3 which Python to link against
ENV PYO3_PYTHON=python3.11

WORKDIR /build

COPY Cargo.toml build.rs ./
COPY src/ ./src/

RUN cargo build --release --bin dvm --bin customer

# Stage 2: Runtime image
# Same base as builder - libpython3.11 guaranteed
FROM python:3.11-slim-bookworm AS runtime

COPY python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --from=builder /build/target/release/dvm     /usr/local/bin/fedstr-dvm
COPY --from=builder /build/target/release/customer /usr/local/bin/fedstr-customer

COPY python/ /opt/fedstr/python/
COPY algorithms/ /opt/fedstr/algorithms/
ENV PYTHONPATH=/opt/fedstr/python

RUN mkdir -p /tmp/fedstr_models
VOLUME /tmp/fedstr_models

CMD ["fedstr-dvm", "--help"]