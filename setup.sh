#!/usr/bin/env bash
# FEDSTR Setup Assistant
# Generates keypairs, configures storage, relay, and FL parameters,
# then writes a ready-to-use .env file.

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Helpers ───────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║           FEDSTR Setup Assistant                 ║${NC}"
    echo -e "${CYAN}${BOLD}║  Decentralized Federated Learning on NOSTR       ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BOLD}${CYAN}── Step $1: $2 ${NC}"
    echo ""
}

print_ok() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_warn() {
    echo -e "  ${YELLOW}⚠${NC}  $1"
}

print_err() {
    echo -e "  ${RED}✗${NC} $1"
}

ask() {
    # ask <prompt> <default>
    # Prompt goes to stderr so it doesn't pollute the captured return value
    local prompt=$1
    local default=$2
    local answer
    echo -en "  ${BOLD}$prompt${NC} [${default}]: " >&2
    read -r answer </dev/tty
    echo "${answer:-$default}"
}

ask_choice() {
    # ask_choice <prompt> <max>
    local prompt=$1
    local max=$2
    local answer
    while true; do
        echo -en "  ${BOLD}$prompt${NC}: " >&2
        read -r answer </dev/tty
        if [[ "$answer" =~ ^[1-9][0-9]*$ ]] && [ "$answer" -ge 1 ] && [ "$answer" -le "$max" ]; then
            echo "$answer"
            return
        fi
        print_err "Please enter a number between 1 and $max"
    done
}

check_python() {
    if command -v python3 &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        print_err "Python not found. Please install Python 3."
        exit 1
    fi
}

generate_hex() {
    local python_bin=$1
    $python_bin -c "import secrets; print(secrets.token_hex(32))"
}

# ── Main ──────────────────────────────────────────────────────────────────────
print_header

# Check .env doesn't already exist
if [ -f ".env" ]; then
    echo -e "  ${YELLOW}A .env file already exists.${NC}"
    echo -en "  Overwrite it? (y/N): " >&2
    read -r overwrite </dev/tty
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo ""
        echo "  Keeping existing .env. Run 'rm .env' to start fresh."
        echo ""
        exit 0
    fi
fi

PYTHON=$(check_python)

# ─────────────────────────────────────────────────────────────────────────────
print_step 1 "Generating DVM keypairs"

echo "  Each DVM needs a unique NOSTR identity (32-byte secret key)."
echo "  These are generated locally and never leave your machine."
echo ""

NUM_KEYS=4
declare -a NSEC_KEYS
for i in $(seq 0 $((NUM_KEYS - 1))); do
    NSEC_KEYS[$i]=$(generate_hex "$PYTHON")
    print_ok "DVM $i: ${NSEC_KEYS[$i]:0:16}...  (truncated for display)"
done

echo ""
print_ok "4 keypairs generated."

# ─────────────────────────────────────────────────────────────────────────────
print_step 2 "Choose storage backend"

echo "  Models trained by DVMs (~400KB each) need a place to be stored"
echo "  between rounds so the customer can download and aggregate them."
echo ""
echo "  [1] Local  — storage container runs on your machine (default)"
echo "               No external server needed. Fully self-contained."
echo ""
echo "  [2] Remote — use the FEDSTR demo server (storage.knikolakakis.com)"
echo "               For paper reviewers / multi-machine demos."
echo "               Requires an auth token — contact: KostisNikolakakis@pm.me"
echo ""

STORAGE_CHOICE=$(ask_choice "Enter choice (1 or 2)" 2)

if [ "$STORAGE_CHOICE" = "1" ]; then
    STORAGE_URL="http://storage:8000"
    STORAGE_AUTH_TOKEN=""
    STORAGE_PROFILE="--profile local"
    print_ok "Local storage selected. Start command: docker compose --profile local up"
else
    echo ""
    echo -en "  ${BOLD}Paste your auth token${NC}: " >&2
    read -r STORAGE_AUTH_TOKEN </dev/tty
    if [ -z "$STORAGE_AUTH_TOKEN" ]; then
        print_warn "No token entered. You can add it later in .env (STORAGE_AUTH_TOKEN=...)."
        STORAGE_AUTH_TOKEN="your_token_here"
    fi
    STORAGE_URL="https://storage.knikolakakis.com"
    STORAGE_PROFILE=""
    print_ok "Remote storage selected."
fi

# ─────────────────────────────────────────────────────────────────────────────
print_step 3 "Choose NOSTR relay"

echo "  The relay is used by DVMs and the customer to exchange events."
echo "  All participants must use the same relay."
echo ""
echo "  [1] wss://relay.knikolakakis.com  (default — stable demo relay)"
echo "  [2] wss://relay.damus.io          (public relay)"
echo "  [3] wss://nos.lol                 (public relay)"
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
print_step 4 "Federated learning parameters"

echo "  Configure the training run. Press Enter to keep defaults."
echo ""

NUM_DVMS=$(ask "Number of DVMs" "2")
ROUNDS=$(ask "Number of rounds" "3")
EPOCHS=$(ask "Epochs per round" "3")
BATCH_SIZE=$(ask "Batch size" "32")
CUSTOMER_WAIT=$(ask "Seconds to wait before customer starts" "30")
DISCOVERY_TIMEOUT=$(ask "DVM discovery timeout (seconds)" "60")

print_ok "Parameters set."

# ─────────────────────────────────────────────────────────────────────────────
print_step 5 "Writing .env"

cat > .env << EOF
# FEDSTR .env — generated by setup.sh
# DO NOT commit this file to git.

# NOSTR relay
NOSTR_RELAY=${NOSTR_RELAY}

# Storage backend
STORAGE_URL=${STORAGE_URL}
STORAGE_AUTH_TOKEN=${STORAGE_AUTH_TOKEN}
STORAGE_PORT=8000

# DVM keypairs (hex secret keys — keep secret)
DVM_0_NSEC_HEX=${NSEC_KEYS[0]}
DVM_1_NSEC_HEX=${NSEC_KEYS[1]}
DVM_2_NSEC_HEX=${NSEC_KEYS[2]}
DVM_3_NSEC_HEX=${NSEC_KEYS[3]}

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

print_ok ".env written successfully."

# ─────────────────────────────────────────────────────────────────────────────
# Ensure .env is in .gitignore
if [ -f ".gitignore" ]; then
    if ! grep -q "^\.env$" .gitignore; then
        echo ".env" >> .gitignore
        print_ok ".env added to .gitignore"
    else
        print_ok ".env already in .gitignore"
    fi
    if ! grep -q "^!\.env\.example$" .gitignore; then
        echo "!.env.example" >> .gitignore
    fi
else
    printf ".env\n!.env.example\n" > .gitignore
    print_ok ".gitignore created"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║              Setup Complete!                     ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Your .env is ready. Next step:${NC}"
echo ""

# Build the correct profiles string based on choices
PROFILES=""
if [ "$STORAGE_CHOICE" = "1" ]; then
    PROFILES="local"
fi
if [ "$NUM_DVMS" = "3" ]; then
    PROFILES="${PROFILES:+$PROFILES,}dvm3"
elif [ "$NUM_DVMS" = "4" ]; then
    PROFILES="${PROFILES:+$PROFILES,}dvm4"
fi

# Build the final command
if [ -n "$PROFILES" ]; then
    COMPOSE_CMD="COMPOSE_PROFILES=$PROFILES docker compose up --build"
else
    COMPOSE_CMD="docker compose up --build"
fi

echo -e "  ${GREEN}${BOLD}${COMPOSE_CMD}${NC}"
echo ""
if [ "$STORAGE_CHOICE" = "1" ]; then
    echo "  This starts: local storage + ${NUM_DVMS} DVMs + customer."
else
    echo "  This starts: ${NUM_DVMS} DVMs + customer (remote storage)."
fi

echo ""
echo "  Other useful commands:"
if [ "$STORAGE_CHOICE" = "1" ]; then
    echo -e "  ${YELLOW}COMPOSE_PROFILES=local,dvm3 docker compose up --build${NC}  # 3 DVMs + local storage"
    echo -e "  ${YELLOW}COMPOSE_PROFILES=local,dvm4 docker compose up --build${NC}  # 4 DVMs + local storage"
else
    echo -e "  ${YELLOW}COMPOSE_PROFILES=dvm3 docker compose up --build${NC}        # 3 DVMs + remote storage"
    echo -e "  ${YELLOW}COMPOSE_PROFILES=dvm4 docker compose up --build${NC}        # 4 DVMs + remote storage"
fi
echo ""
echo "  To watch logs per service:"
echo -e "  ${YELLOW}docker compose logs -f customer${NC}"
echo -e "  ${YELLOW}docker compose logs -f dvm-0${NC}"
echo ""
