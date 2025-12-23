# FEDSTR: Federated Learning on the NOSTR

**A Decentralized Marketplace for Federated Learning**


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Overview

FEDSTR (Federated Learning on NOSTR) is a proof-of-concept implementation of a **decentralized federated learning marketplace** built on the [NOSTR protocol](https://github.com/nostr-protocol/nostr). 

Unlike traditional federated learning systems that rely on centralized coordinators, FEDSTR enables:


- 🌐 **True decentralization** - Communication over public NOSTR relays
- ⚡ **Payment integration** - Built-in Lightning Network payment protocol (NIP-57)
- 🔐 **Privacy-preserving** - Encrypted coordination and message exchange 
- 📊 **Validation** - Algorithmic detection of malicious or lazy service providers
- 🔒 **Cryptographic integrity verification** - SHA-256 hash verification ensures models cannot be tampered

**Paper:** [FEDSTR: A Decentralized Marketplace for Federated Learning on NOSTR](link-to-paper)

---

## Key Features

### Protocol Implementation
- ✅ **NIP-90 Data Vending Machines** - Job request/result protocol
- ✅ **Custom Event Kinds** - Kind 8000 (JobRequest), 6000 (JobResult), 7000 (JobFeedback)
- ✅ **Multi-relay communication** - Redundant relay connections for reliability
- ✅ **Event deduplication** - Automatic handling of duplicate events across relays

### Cryptographic Security
- ✅ **SHA-256 model integrity** - Every model exchange is hash-verified
- ✅ **Schnorr signatures** - All events cryptographically signed
- ✅ **Zero tampering** - 100% hash verification success in testing 

### Federated Learning
- ✅ **FedAvg algorithm** - Classical federated averaging
- 🔄 **Validation mechanisms** - Detect malicious/lazy service providers
- ✅ **Multi-round training** - Support for iterative model improvement
- ✅ **Data splitting** - Automatic dataset partitioning across DVMs

### Storage Options
- ✅ **Local storage** - File-based model storage (default)
- ✅ **HTTP server support** - Remote model storage
- 🔄 **Cloud storage** - Blossom/IPFS integration (planned)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NOSTR Relays (Public)                     │
│              relay.damus.io, nos.lol, etc.                   │
└─────────────────────────────────────────────────────────────┘
                          ↑     ↓
                    WebSocket Events
                          ↑     ↓
        ┌─────────────────┴─────┴──────────────────┐
        │                                          │
   ┌────▼────┐                               ┌─────▼─────┐
   │ Customer│                               │    DVM    │
   │(Aggregator)                             │ (Service  │
   │         │                               │ Provider) │
   └────┬────┘                               └─────┬─────┘
        │                                          │
        │ 1. JobRequest (Kind 8000)                │
        │ ────────────────────────────────────────>│
        │                                          │
        │ 2. JobFeedback (Kind 7000)               │
        │ <────────────────────────────────────────│
        │                                          │
        │                                  ┌───────▼────────┐
        │                                  │ Train on data  │
        │                                  │ Hash: SHA-256  │
        │                                  └───────┬────────┘
        │                                          │
        │ 3. JobResult (Kind 6000) + Hash          │
        │ <────────────────────────────────────────│
        │                                          │
   ┌────▼────────┐                                 │
   │Verify Hash  │                                 │
   │Validate     │                                 │
   │FedAvg       │                                 │
   └────┬────────┘                                 │
        │                                          │
        │ 4. Payment (NIP-57)                      │
        │ ────────────────────────────────────────>│
        │                                          │
```

---

## Quick Start

### Prerequisites

- **Rust** 1.70+ ([Install Rust](https://rustup.rs/))
- **Python** 3.8+ with PyTorch
- **Conda** (recommended) or virtualenv

### Complete Installation Guide

#### Step 1: Install Rust
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

#### Step 2: Install Conda
```bash
# Download Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# For macOS:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
# bash Miniconda3-latest-MacOSX-x86_64.sh

# Follow prompts, then reload shell
source ~/.bashrc  # or source ~/.zshrc for macOS
```

#### Step 3: Clone Repository
```bash
git clone https://github.com/yourusername/fedstr-poc
cd fedstr-poc
```

#### Step 4: Setup Python Environment
```bash
# Create conda environment with Python 3.9
conda create -n fedstr python=3.9 -y
conda activate fedstr

# Install PyTorch and dependencies
conda install pytorch torchvision -c pytorch -y

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

### Step 5: Set Environment Variables

```bash
# Set CONDA_PREFIX for Rust build
export CONDA_PREFIX=$HOME/miniconda3/envs/fedstr

# Platform-specific library path configuration
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_FALLBACK_LIBRARY_PATH"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# Add to your shell config for persistence
echo "export CONDA_PREFIX=$HOME/miniconda3/envs/fedstr" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc  # Linux
# or for macOS: echo 'export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_FALLBACK_LIBRARY_PATH"' >> ~/.zshrc
```

#### Step 6: Build the Project
```bash
# Build in release mode
cargo build --release

# This will:
# ✓ Compile Rust code
# ✓ Link PyO3 bindings to your conda Python
# ✓ Create binaries in target/release/
```


#### Step 7: Verify Installation
```bash
# Make sure conda environment is active
conda activate fedstr

# Run tests
cargo test

# Expected output:
# running 15 tests
# test ml::validation::tests::... ok
# test protocol::job_request::tests::... ok
# ...
# test result: ok. 15 passed; 0 failed
```

#### Troubleshooting

**Issue: PyO3 linking errors**
```bash
# Ensure CONDA_PREFIX is set correctly
echo $CONDA_PREFIX
# Should output: /path/to/miniconda3/envs/fedstr

# If empty, set it:
export CONDA_PREFIX=$(conda info --base)/envs/fedstr

# Clean and rebuild
cargo clean
cargo build --release
```

**Issue: Python not found**
```bash
# Verify conda environment
conda activate fedstr
which python
# Should output: /path/to/miniconda3/envs/fedstr/bin/python

# Verify PyTorch
python -c "import torch; print(torch.__version__)"
```

**Issue: MNIST download fails**
```bash
# Manual download (if needed)
mkdir -p data/MNIST/raw
cd data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
cd ../../..
```

### Run A Federated Learning Session

**Terminal 0 - File Server**

```bash
python python/file_server.py --port 8000
```

**Terminal 1 - Start DVM 0**
```bash
./target/release/dvm --id 0 --storage http://localhost:8000
```

**Terminal 2 - Start DVM 1**
```bash
./target/release/dvm --id 1 --storage http://localhost:8000
```

**Terminal 3 - Run Customer (Coordinator)**
```bash
./target/release/customer \
  --num-dvms 2 \
  --rounds 3 \
  --storage http://localhost:8000 \
  --dvms npub1wx386...,npub1ll2xc...
```

Or

**Terminal 1 - Start DVM 0:**
```bash
./target/release/dvm --id 0
```

**Terminal 2 - Start DVM 1:**
```bash
./target/release/dvm --id 1
```

**Terminal 3 - Run Customer (Coordinator):**
```bash
./target/release/customer \
  --num-dvms 2 \
  --rounds 3 \
  --dvms npub1wx386...,npub1ll2xc...
```

**Expected Output:**
```
✓ Connected to relays
✓ Hash verified - Model integrity confirmed! 
✓ Validation passed 
```

---

## Usage Examples

### Basic Training (2 DVMs, 3 Rounds)

```bash
# Terminal 1: DVM 0
./target/release/dvm --id 0

# Terminal 2: DVM 1  
./target/release/dvm --id 1

# Terminal 3: Customer
./target/release/customer \
  --num-dvms 2 \
  --rounds 3 \
  --dvms npub1wx3863kca6k2lmgm24qm6c303z28xa5f20kw23mp3nn8wzyahqmqaycgkq,npub1ll2xc6kwwllsuuwwzw8s05589m3rjdk2upvzn2ve5l9m0jgnke7sxnddsr
```


---

## How It Works

### 1. Customer Publishes JobRequest (Kind 8000)

```json
{
  "kind": 8000,
  "tags": [
    ["job_id", "abc123"],
    ["dataset", "mnist"],
    ["data_split", "0-30000"],
    ["model", "tiny_linear"],
    ["round", "1"],
    ["p", "npub1wx3863..."]  // Target DVM
  ]
}
```

### 2. DVM Trains & Computes Hash

```rust
// Train model
let trained_model = train_on_data(data_split);

// Serialize to bytes
let model_bytes = serialize(trained_model);

// Compute SHA-256 hash
let hash = sha256(model_bytes);
// hash = "a1d51c8ef1c556a8992b466785207cd0859459e84f2aded04a09440fca530292"
```

### 3. DVM Publishes JobResult (Kind 6000)

```json
{
  "kind": 6000,
  "tags": [
    ["job_id", "abc123"],
    ["e", "<job-request-event-id>"],
    ["model_url", "file:///tmp/fedstr_models/model_a1d51c8e.bin"],
    ["model_hash", "a1d51c8ef1c556a8992b466785207cd0859459e84f2aded04a09440fca530292"],
    ["loss", "0.3003"],
    ["accuracy", "0.9117"]
  ]
}
```

### 4. Customer Verifies Hash

```rust
// Download model
let downloaded_bytes = download(model_url);

// Recompute hash
let computed_hash = sha256(downloaded_bytes);

// Verify
if computed_hash == expected_hash {
    println!("✅ Hash verified - Model integrity confirmed!");
} else {
    println!("❌ Hash mismatch - Model tampered!");
}
```

### 5. Customer Validates & Aggregates

```rust
// Validate output (Algorithm 3)
if validate_output(model, validation_dataset) {
    // Aggregate with FedAvg
    let global_model = fedavg([model_dvm0, model_dvm1]);
    
    // Prepare for next round
    publish_job_request(round + 1, global_model);
}
```

---

## Project Structure

```
fedstr-poc/
├── src/
│   ├── bin/
│   │   ├── customer.rs       # Customer/aggregator binary
│   │   └── dvm.rs            # DVM service provider binary
│   ├── protocol/
│   │   ├── mod.rs               # Protocol module exports
│   │   ├── job_request.rs       # JobRequest event (Kind 8000)
│   │   ├── job_result.rs        # JobResult event (Kind 6000)
│   │   ├── job_feedback.rs      # JobFeedback event (Kind 7000)
│   │   └── payments.rs          # Payment protocol (NIP-57, dummy)
│   ├── ml/
│   │   ├── training.rs       # PyTorch training interface
|   |   ├── mod.rs            # ML module exports
│   │   ├── aggregation.rs    # FedAvg aggregation
│   │   └── validation.rs     # Output validation 
│   ├── storage.rs            # Model storage backends
│   └── lib.rs                # Library exports
├── python/
│   ├── train_tiny.py           # PyTorch training implementation
│   └── data/                  # Dataset
├── Cargo.toml               # Rust dependencies
├── build.rs                     # Cargo build script (PyO3 configuration)
├── Cargo.lock                   # Locked dependency versions
├── file_server.py               # HTTP model storage server (for testing)
├── README.md                # This file
└── paper/
    └── FEDSTR_paper.pdf     # Research paper
```

---

## Configuration

### Environment Variables

```bash
# Storage backend
export STORAGE_BACKEND=local              # local | http | blossom
export STORAGE_PATH=/tmp/fedstr_models    # For local storage

# NOSTR relays (comma-separated)
export NOSTR_RELAYS=wss://relay.damus.io,wss://nos.lol

# Blossom server (if using blossom storage)
export BLOSSOM_SERVER=https://blossom.azzamo.net

# Python environment
export CONDA_PREFIX=/path/to/conda/env
```

### Command Line Options

#### Customer (Aggregator)

```bash
./target/release/customer --help

Options:
  --num-dvms <N>              Number of DVMs to use [default: 2]
  --rounds <N>                Number of training rounds [default: 3]
  --dataset <NAME>            Dataset name [default: mnist]
  --model <NAME>              Model architecture [default: tiny_linear]
  --dvms <NPUBS>              Comma-separated DVM public keys (npub format)
  --relays <URLS>             Comma-separated relay URLs
  --epochs <N>                Epochs per round [default: 1]
  --batch-size <N>            Batch size [default: 32]
```

#### DVM (Service Provider)

```bash
./target/release/dvm --help

Options:
  --id <N>                    DVM identifier [default: 0]
  --storage <BACKEND>         Storage backend: local | http://... | blossom [default: local]
  --relays <URLS>             Comma-separated relay URLs
  --keypair-file <PATH>       Path to saved keypair (for persistent identity)
```

---

## Development

### Build from Source

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run with logging
RUST_LOG=debug ./target/release/customer ...
```

### Python Development

```bash
# Activate conda environment
conda activate fedstr

# Install development dependencies
pip install pytest black flake8

# Run Python tests
pytest python/tests/

# Format code
black python/
```

### Code Style

```bash
# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy

# Format Python code
black python/
```

---

## Security

### Cryptographic Guarantees

✅ **Model Integrity:** SHA-256 hash verification prevents tampering  
✅ **Event Authenticity:** Schnorr signatures verify event origin  
✅ **Relay Independence:** Multiple relays prevent single point of failure  
✅ **No Trusted Parties:** Fully decentralized architecture

### Validation Mechanisms

✅ **Algorithm 3 (Paper):** Detects lazy/malicious DVMs  
✅ **Loss Monitoring:** Validates training progress  
✅ **Peer Comparison:** Cross-validates DVM outputs  
✅ **Timeout Protection:** Prevents DoS by slow DVMs

### Known Limitations

⚠️ **Sybil Attacks:** No reputation system (future work)  
⚠️ **Data Poisoning:** No defense against malicious training data  
⚠️ **Payment Security:** Dummy implementation


### Validation and Lazy DVM Detection

**Current Implementation (PoC):**
The validation of non-lazy DVMs in this proof-of-concept is simplified and relies on the computational results reported by the DVMs themselves (training metrics like loss and accuracy). The `validation.rs` module performs basic checks:
- Verifies loss improvement over previous rounds
- Compares performance against peer DVMs
- Checks minimum accuracy thresholds

**Limitation:**
This approach assumes DVMs honestly report their training metrics. A malicious DVM could:
- Report fake metrics without actually training
- Return a slightly modified previous model
- Claim training completion while doing minimal work

**Production Recommendation:**
For robust detection of lazy or malicious DVMs, the customer (aggregator) should:

1. **Independent Validation Set:** Maintain a private validation dataset

2. **Cross-Validation:** Run inference on received models before aggregation

3. **Statistical Tests:** Compare model parameter distributions

4. **Challenge-Response:** Occasionally request proof of computation (Ask DVM to provide intermediate gradients or activations)

**Future Work:**
- Reputation system based on validation history
- Economic penalties for lazy/malicious behavior

---

## Troubleshooting

### Common Issues

**Issue:** `Error connecting to relay`
```bash
# Solution: Check relay status
curl -I https://relay.damus.io

# Try different relays
./target/release/dvm --relays wss://nos.lol
```

**Issue:** `Hash verification failed`
```bash
# This should never happen! Indicates:
# - Relay tampering (very unlikely)
# - Storage corruption
# - Network transmission error

# Check logs:
RUST_LOG=debug ./target/release/customer ...
```

**Issue:** `PyTorch import error`
```bash
# Ensure conda environment is activated
conda activate fedstr

# Reinstall PyTorch
pip install torch torchvision --force-reinstall
```

**Issue:** `DVM not receiving JobRequests`
```bash
# Check DVM public key matches customer command
./target/release/dvm --id 0  # Prints: "Pubkey (npub): npub1wx3863..."

# Use this exact npub in customer command:
./target/release/customer --dvms npub1wx3863...
```

---

## Documentation

- **Paper:** [FEDSTR: A Decentralized Marketplace for Federated Learning on NOSTR](link-to-paper)
- **NOSTR Protocol:** [github.com/nostr-protocol/nostr](https://github.com/nostr-protocol/nostr)
- **NIP-90 (Data Vending Machines):** [NIP-90 Spec](https://github.com/nostr-protocol/nips/blob/master/90.md)
- **NIP-57 (Lightning Payments):** [NIP-57 Spec](https://github.com/nostr-protocol/nips/blob/master/57.md)
- **FedAvg Paper:** [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

---

### Future Work

- ⚡ **Payments:** Full Lightning Network integration
- 🌐 **Storage:** Blossom/IPFS integration
- 📱 **UX:** Web/mobile client development
- 📊 **ML:** Support for more algorithms (DiLoCo, SecAgg, etc.)

### Custom Dataset

```bash
./target/release/customer \
  --num-dvms 3 \
  --rounds 5 \
  --dataset fashion-mnist \
  --dvms npub1...,npub2...,npub3...
```

### Custom Model Storage

```bash
# Local path
./target/release/dvm --id 0 --storage /my/custom/path

# HTTP server
./target/release/dvm --id 0 --storage http://my-server:8081

# Blossom (future)
./target/release/dvm --id 0 --storage blossom
```

### Custom Relays

```bash
./target/release/dvm --id 0 \
  --relays wss://relay.damus.io,wss://nos.lol,wss://relay.nostr.band
```

---

## Citation

If you use FEDSTR in your research, please cite:

```bibtex
@article{nikolakakis2024fedstr,
  title={FEDSTR: A Decentralized Marketplace for Federated Learning on NOSTR},
  author={Nikolakakis, Konstantinos E. and Chantzialexiou, George and Kalogerias, Dionysis},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

- **Konstantinos E. Nikolakakis** - [KostisNikolakakis@pm.me](mailto:KostisNikolakakis@pm.me)
- **George Chantzialexiou** - [george.chantzialexiou@gmail.com](mailto:george.chantzialexiou@gmail.com)
- **Dionysis Kalogerias** - [dionysis.kalogerias@yale.edu](mailto:dionysis.kalogerias@yale.edu)

---

## Acknowledgments

- NOSTR protocol developers
- Rust community
- Bitcoin & Lightning Network developers

---

## Contact

- **Email:** KostisNikolakakis@pm.me
- **NOSTR:** npub1... (add your npub)

---

**Built with ❤️ for decentralized AI**
