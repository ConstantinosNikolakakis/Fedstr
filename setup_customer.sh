#!/usr/bin/env bash
# FEDSTR Customer Setup
# Configures the customer/coordinator machine.
# The customer controls all training configuration — DVMs just execute.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║      FEDSTR Customer Setup Assistant             ║${NC}"
    echo -e "${CYAN}${BOLD}║   Configure the training coordinator             ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BOLD}${CYAN}── Step $1: $2 ${NC}"
    echo ""
}

print_ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
print_warn() { echo -e "  ${YELLOW}⚠${NC}  $1"; }
print_err()  { echo -e "  ${RED}✗${NC} $1"; }

ask() {
    local prompt=$1 default=$2 answer
    echo -en "  ${BOLD}$prompt${NC} [${default}]: " >&2
    read -r answer </dev/tty
    echo "${answer:-$default}"
}

ask_choice() {
    local prompt=$1 max=$2 answer
    while true; do
        echo -en "  ${BOLD}$prompt${NC}: " >&2
        read -r answer </dev/tty
        if [[ "$answer" =~ ^[1-9][0-9]*$ ]] && [ "$answer" -ge 1 ] && [ "$answer" -le "$max" ]; then
            echo "$answer"; return
        fi
        print_err "Please enter a number between 1 and $max"
    done
}

check_python() {
    if command -v python3 &>/dev/null; then echo "python3"
    elif command -v python &>/dev/null; then echo "python"
    else print_err "Python not found."; exit 1; fi
}

generate_hex() { $1 -c "import secrets; print(secrets.token_hex(32))"; }

# ── Main ──────────────────────────────────────────────────────────────────────
print_header

echo "  The customer is the training coordinator. It:"
echo "    • Selects the algorithm, model, and dataset"
echo "    • Manages outer optimization (Nesterov momentum for DiLoCo)"
echo "    • Discovers DVMs and distributes training jobs"
echo "    • Aggregates results after each round"
echo ""

if [ -f ".env" ]; then
    echo -e "  ${YELLOW}A .env file already exists.${NC}"
    echo -en "  Overwrite it? (y/N): " >&2
    read -r overwrite </dev/tty
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo "  Keeping existing .env."
        exit 0
    fi
fi

PYTHON=$(check_python)

# ─────────────────────────────────────────────────────────────────────────────
print_step 1 "Choose FL algorithm"

echo "  [1] FedAvg  — Federated Averaging (McMahan et al., 2017)"
echo "               SGD inner optimizer, weighted average outer step."
echo "               Fast baseline. Good for MNIST benchmarks."
echo ""
echo "  [2] DiLoCo  — Distributed Low-Communication Training (Douillard et al., 2023)"
echo "               AdamW inner optimizer, Nesterov momentum outer step."
echo "               Designed for LLM training across poorly connected nodes."
echo ""

ALGO_CHOICE=$(ask_choice "Enter choice (1 or 2)" 2)

if [ "$ALGO_CHOICE" = "1" ]; then
    ALGORITHM="fedavg"
    DATASET="mnist"
    DEFAULT_EPOCHS="3"
    DEFAULT_ROUNDS="3"
    print_ok "FedAvg selected."
else
    ALGORITHM="diloco"
    DATASET="shakespeare"
    DEFAULT_EPOCHS="500"
    DEFAULT_ROUNDS="10"
    print_ok "DiLoCo selected."
    echo ""
    print_warn "DiLoCo uses H=500 inner steps/round (AdamW) and Nesterov outer."
    print_warn "Recommended: 500 inner steps, 10-30 rounds for Shakespeare."
    print_warn "Each round ~15-20 min on CPU, ~2-3 min on GPU."
fi

# ─────────────────────────────────────────────────────────────────────────────
print_step 2 "Storage backend"

echo "  Trained model checkpoints are stored between rounds."
echo ""
echo "  [1] Remote — storage.knikolakakis.com (demo server)"
echo "               For multi-machine runs and paper reviewers."
echo "               Requires auth token — contact: KostisNikolakakis@pm.me"
echo ""
echo "  [2] Local  — storage container on this machine"
echo "               Fully self-contained. Good for single-machine testing."
echo ""

STORAGE_CHOICE=$(ask_choice "Enter choice (1 or 2)" 2)

if [ "$STORAGE_CHOICE" = "1" ]; then
    echo ""
    echo -en "  ${BOLD}Paste your auth token${NC}: " >&2
    read -r STORAGE_AUTH_TOKEN </dev/tty
    if [ -z "$STORAGE_AUTH_TOKEN" ]; then
        print_warn "No token entered. Add later: STORAGE_AUTH_TOKEN=..."
        STORAGE_AUTH_TOKEN="your_token_here"
    fi
    STORAGE_URL="https://storage.knikolakakis.com"
    STORAGE_PROFILE=""
    print_ok "Remote storage selected."
else
    STORAGE_URL="http://storage:8000"
    STORAGE_AUTH_TOKEN=""
    STORAGE_PROFILE="local"
    print_ok "Local storage selected."
fi

# ─────────────────────────────────────────────────────────────────────────────
print_step 3 "NOSTR relay"

echo "  The relay coordinates communication between customer and DVMs."
echo "  All participants must use the same relay."
echo ""
echo "  [1] wss://relay.knikolakakis.com  (default — stable demo relay)"
echo "  [2] wss://relay.damus.io"
echo "  [3] wss://nos.lol"
echo "  [4] Custom"
echo ""

RELAY_CHOICE=$(ask_choice "Enter choice (1-4)" 4)
NOSTR_RELAY="wss://relay.knikolakakis.com"
case $RELAY_CHOICE in
    1) NOSTR_RELAY="wss://relay.knikolakakis.com" ;;
    2) NOSTR_RELAY="wss://relay.damus.io" ;;
    3) NOSTR_RELAY="wss://nos.lol" ;;
    4)
        echo -en "  ${BOLD}Enter relay URL${NC}: " >&2
        read -r NOSTR_RELAY </dev/tty
        ;;
esac
print_ok "Relay: $NOSTR_RELAY"

# ─────────────────────────────────────────────────────────────────────────────
print_step 4 "Generate customer keypair and DVM keys"

echo "  Generating customer identity and local DVM keypairs..."
echo ""

CUSTOMER_NSEC=$(generate_hex "$PYTHON")
DVM_0_NSEC=$(generate_hex "$PYTHON")
DVM_1_NSEC=$(generate_hex "$PYTHON")
DVM_2_NSEC=$(generate_hex "$PYTHON")
DVM_3_NSEC=$(generate_hex "$PYTHON")

print_ok "Customer keypair generated."
print_ok "4 DVM keypairs generated (for local DVMs if needed)."

# ─────────────────────────────────────────────────────────────────────────────
print_step 5 "Training parameters"

echo "  Configure the federated learning run."
echo "  These values are sent to DVMs via job requests."
echo ""

NUM_DVMS=$(ask "Number of DVMs to use" "2")
ROUNDS=$(ask "Number of outer rounds" "$DEFAULT_ROUNDS")
EPOCHS=$(ask "Inner steps per round (H)" "$DEFAULT_EPOCHS")
BATCH_SIZE=$(ask "Batch size" "32")
CUSTOMER_WAIT=$(ask "Seconds to wait before customer starts" "30")
DISCOVERY_TIMEOUT=$(ask "DVM discovery timeout (seconds)" "60")

print_ok "Training parameters set."

# ─────────────────────────────────────────────────────────────────────────────
print_step 6 "Write .env"

cat > .env << EOF
# FEDSTR Customer .env — generated by setup_customer.sh
# DO NOT commit this file to git.

# NOSTR relay
NOSTR_RELAY=${NOSTR_RELAY}

# Storage backend
STORAGE_URL=${STORAGE_URL}
STORAGE_AUTH_TOKEN=${STORAGE_AUTH_TOKEN}
STORAGE_PORT=8000

# Algorithm and dataset
ALGORITHM=${ALGORITHM}
DATASET=${DATASET}

# DVM keypairs (for local DVMs)
DVM_0_NSEC_HEX=${DVM_0_NSEC}
DVM_1_NSEC_HEX=${DVM_1_NSEC}
DVM_2_NSEC_HEX=${DVM_2_NSEC}
DVM_3_NSEC_HEX=${DVM_3_NSEC}

# Federated learning parameters
NUM_DVMS=${NUM_DVMS}
ROUNDS=${ROUNDS}
EPOCHS_PER_ROUND=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
CUSTOMER_WAIT=${CUSTOMER_WAIT}
DISCOVERY_TIMEOUT=${DISCOVERY_TIMEOUT}

# Logging
RUST_LOG=info
EOF

print_ok ".env written."

# Ensure .gitignore
if [ -f ".gitignore" ]; then
    grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
else
    printf ".env\n!.env.example\n" > .gitignore
fi
print_ok ".env added to .gitignore"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║           Customer Setup Complete!               ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Start training:${NC}"
echo ""

# Build profiles string
PROFILES=""
[ "$STORAGE_CHOICE" = "2" ] && PROFILES="local"
[ "$NUM_DVMS" = "3" ] && PROFILES="${PROFILES:+$PROFILES,}dvm3"
[ "$NUM_DVMS" = "4" ] && PROFILES="${PROFILES:+$PROFILES,}dvm4"

if [ -n "$PROFILES" ]; then
    echo -e "  ${GREEN}${BOLD}COMPOSE_PROFILES=$PROFILES docker compose up --build${NC}"
else
    echo -e "  ${GREEN}${BOLD}docker compose up --build${NC}"
fi

echo ""
echo "  This starts: ${NUM_DVMS} local DVMs + customer [${ALGORITHM}]"
echo ""
echo -e "  ${BOLD}Or use external DVMs only (no local DVMs):${NC}"
echo -e "  ${YELLOW}docker compose up customer --build${NC}"
echo ""
echo "  External DVMs auto-discovered via relay — no manual config needed."
echo ""
echo -e "  ${BOLD}Share with DVM operators:${NC}"
echo -e "  ${YELLOW}Relay: ${NOSTR_RELAY}${NC}"
echo -e "  ${YELLOW}Algorithm: ${ALGORITHM} | Dataset: ${DATASET}${NC}"
echo ""
echo -e "  ${BOLD}Good luck! 🚀${NC}"
echo ""
