# FEDSTR — DiLoCo + NanoGPT Branch

**Branch:** `feature/diloco-nanogpt`  
**Base:** `docker` branch  
**Status:** Active development — not yet merged to main

This branch extends FEDSTR with the DiLoCo algorithm (Douillard et al., 2023) and a
nanoGPT-based language model trained on the Tiny Shakespeare dataset.

---

## What's new in this branch

| Feature | Description |
|---------|-------------|
| **DiLoCo algorithm** | AdamW inner optimizer + Nesterov momentum outer step |
| **Persistent momentum** | `v_t` state uploaded to storage and reused across outer rounds |
| **nanoGPT model** | 4L 4H 128d transformer, ~1.5M params, character-level |
| **Shakespeare dataset** | Tiny Shakespeare (~1MB), split across DVMs |
| **JobRequest carries full config** | Algorithm, model arch, hyperparams — DVMs need no env vars |
| **AlgorithmConfig** | `algorithms/{algo}/config.yaml` is single source of truth |
| **Training logs** | Per-step ML logs (`logs/dvm_N_training.log`) |
| **Protocol logs** | Full protocol output (`logs/dvm_N_protocol.log`, `logs/customer_protocol.log`) |
| **Three setup scripts** | `setup_deploy_all.sh`, `setup_customer.sh`, `setup_dvm.sh` |

---

## Quick Start (all-in-one, single machine)

```bash
git clone -b feature/diloco-nanogpt https://github.com/ConstantinosNikolakakis/Fedstr.git
cd Fedstr

# Prepare Shakespeare dataset (once)
python3 algorithms/diloco/prepare_data.py

# Configure and run
./setup_deploy_all.sh
docker compose up --build
```

---

## Setup Scripts

### `setup_deploy_all.sh` — All-in-one (customer + DVMs on same machine)
Best for: local testing, single-machine demos.

```bash
./setup_deploy_all.sh
docker compose up --build
```

### `setup_customer.sh` — Customer only
Best for: distributed deployment where DVMs run on separate machines.

```bash
git clone -b feature/diloco-nanogpt https://github.com/ConstantinosNikolakakis/Fedstr.git
cd Fedstr
./setup_customer.sh
docker compose up customer --build
```

### `setup_dvm.sh` — DVM only
Best for: a machine that will only provide compute.

```bash
git clone -b feature/diloco-nanogpt https://github.com/ConstantinosNikolakakis/Fedstr.git
cd Fedstr
./setup_dvm.sh
python3 algorithms/diloco/prepare_data.py
docker compose up dvm-0 --build
```

---

## Algorithm Directory Structure

```
algorithms/
├── fedavg/
│   ├── train.py        # SGD inner optimization on MNIST
│   ├── aggregate.py    # Weighted average outer step
│   └── config.yaml     # FedAvg hyperparameters
└── diloco/
    ├── model.py        # NanoGPT architecture (Karpathy 2022)
    ├── train.py        # AdamW inner optimization on Shakespeare
    ├── aggregate.py    # Nesterov momentum outer step + v_t persistence
    ├── config.yaml     # DiLoCo hyperparameters
    └── prepare_data.py # Download and tokenize Shakespeare dataset
```

---

## DiLoCo Implementation

Implements Algorithm 1 from Douillard et al. (2023):

- **Inner optimizer**: AdamW (`lr=0.001`, `weight_decay=0.1`)
- **Outer optimizer**: Nesterov momentum (`lr_outer=0.7`, `momentum=0.9`)
- **Momentum state** (`v_t`) persisted across outer rounds via storage server
- **Data sharding**: each DVM trains on a contiguous fraction of Shakespeare

---

## Architecture: JobRequest as Single Source of Truth

A key architectural change in this branch: **the customer embeds all training
configuration in the JobRequest**. DVMs need no algorithm-specific environment
variables — they receive everything needed to execute a training job from the
job request itself.

```
Customer reads:           Sends to DVMs via JobRequest:
  .env (ALGORITHM)    →     algorithm, dataset, model_type
  config.yaml         →     n_layer, n_head, n_embd, block_size
                      →     lr_inner, weight_decay, grad_clip
                      →     epochs (H), batch_size
                      →     current_model_params (URL + hash)
```

This enables true marketplace semantics: DVMs are commodity compute providers
that can serve any customer without prior configuration.

---

## Logs

Each run produces 6 log files in `logs/`:

| File | Contents |
|------|----------|
| `dvm_0_training.log` | Per-step loss, LR, val PPL for DVM 0 |
| `dvm_1_training.log` | Per-step loss, LR, val PPL for DVM 1 |
| `customer_training.log` | Round summaries: train loss, val loss, val PPL |
| `dvm_0_protocol.log` | Full protocol output for DVM 0 |
| `dvm_1_protocol.log` | Full protocol output for DVM 1 |
| `customer_protocol.log` | Full protocol output for customer |

---
