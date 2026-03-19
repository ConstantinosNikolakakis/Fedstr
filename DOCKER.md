# FEDSTR — Docker Quickstart

Run a complete federated learning demo (2 DVMs, 3 rounds) in one command.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/install/) (v2+)
- That's it. No Rust, no Python, no conda.

## Run

```bash
# 1. Clone
git clone https://github.com/ConstantinosNikolakakis/Fedstr.git
cd Fedstr

# 2. Configure (optional — defaults work out of the box)
cp .env.example .env
# edit .env to change relay, num DVMs, rounds, etc.

# 3. Start
docker compose up
```

Watch the customer coordinate training rounds across DVMs with hash-verified
model exchange. Ctrl-C when done, then:

```bash
docker compose down -v   # clean up
```

## Scale DVMs

```bash
# Run 4 DVMs instead of the default 2
NUM_DVMS=4 docker compose up --scale dvm=4

# Or edit .env:  NUM_DVMS=4
```

## Use a public NOSTR relay

Edit `.env`:
```
NOSTR_RELAY=wss://relay.damus.io
```

## What's running

| Service   | What it does                                          |
|-----------|-------------------------------------------------------|
| `storage` | HTTP file server for model parameter blobs (port 8000)|
| `dvm`     | NOSTR service provider — trains local model partition |
| `customer`| NOSTR coordinator — splits data, aggregates, validates|

All services share a Docker volume for model files. The customer starts
10 seconds after the DVMs so they have time to publish discoverability events.

## Local dev (without Docker)

If you prefer to build natively with your own conda environment:

```bash
# Rust + Python (conda or system)
export CONDA_PREFIX=/path/to/your/conda/env   # optional
cargo build --release --bin dvm --bin customer

# Start storage server
python3 file_server.py --port 8000 &

# Start DVMs (in separate terminals)
./target/release/fedstr-dvm --id 0 --storage http://localhost:8000
./target/release/fedstr-dvm --id 1 --storage http://localhost:8000

# Start customer
./target/release/fedstr-customer --num-dvms 2 --rounds 3 --storage http://localhost:8000
```
