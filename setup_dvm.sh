#!/usr/bin/env bash
# FEDSTR DVM Setup
# Configures a machine to run as a FEDSTR Service Provider (DVM).
# DVMs are commodity compute workers — they receive full training config
# from the customer via job requests. No algorithm/dataset config needed here.

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
    echo -e "${CYAN}${BOLD}║        FEDSTR DVM Setup Assistant                ║${NC}"
    echo -e "${CYAN}${BOLD}║   Configure this machine as a compute provider   ║${NC}"
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
print_step 1 "How many DVMs on this machine?"

echo "  Each DVM is a separate worker process with its own NOSTR identity."
echo "  You can run multiple DVMs on one machine if you have enough resources."
echo ""
echo "  [1] 1 DVM"
echo "  [2] 2 DVMs"
echo "  [3] 3 DVMs"
echo "  [4] 4 DVMs"
echo ""

NUM_DVMS=$(ask_choice "Enter choice (1-4)" 4)
print_ok "${NUM_DVMS} DVM(s) will be configured."

# ─────────────────────────────────────────────────────────────────────────────
print_step 2 "Generate DVM keypairs"

echo "  Each DVM needs a unique NOSTR identity."
echo "  Generated locally — never leave this machine."
echo ""

declare -a NSEC_KEYS
for i in $(seq 0 $((NUM_DVMS - 1))); do
    NSEC_KEYS[$i]=$(generate_hex "$PYTHON")
    print_ok "DVM $i keypair generated: ${NSEC_KEYS[$i]:0:16}... (truncated)"
done

# ─────────────────────────────────────────────────────────────────────────────
print_step 3 "Storage backend"

echo "  DVMs upload trained model checkpoints after each inner round."
echo ""
echo "  [1] Remote — storage.knikolakakis.com (demo server)"
echo "               Requires auth token — contact: KostisNikolakakis@pm.me"
echo ""
echo "  [2] Local  — storage container on this machine"
echo "               Fully self-contained. Use with --profile local."
echo ""

STORAGE_CHOICE=$(ask_choice "Enter choice (1 or 2)" 2)

if [ "$STORAGE_CHOICE" = "1" ]; then
    echo ""
    echo -en "  ${BOLD}Paste your auth token${NC}: " >&2
    read -r STORAGE_AUTH_TOKEN </dev/tty
    if [ -z "$STORAGE_AUTH_TOKEN" ]; then
        print_warn "No token entered. Add it later: STORAGE_AUTH_TOKEN=..."
        STORAGE_AUTH_TOKEN="your_token_here"
    fi
    STORAGE_URL="https://storage.knikolakakis.com"
    STORAGE_PROFILE=""
    print_ok "Remote storage selected."
else
    STORAGE_URL="http://storage:8000"
    STORAGE_AUTH_TOKEN=""
    STORAGE_PROFILE="--profile local"
    print_ok "Local storage selected."
fi

# ─────────────────────────────────────────────────────────────────────────────
print_step 4 "NOSTR relay"

echo "  DVMs listen for job requests on this relay."
echo "  Must match the relay used by the customer."
echo ""
echo "  [1] wss://relay.knikolakakis.com  (default)"
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
print_step 5 "Write .env"

cat > .env << EOF
# FEDSTR DVM .env — generated by setup_dvm.sh
# DO NOT commit this file to git.
# Note: Algorithm/dataset/model config comes from the customer via job requests.
# DVMs only need credentials and relay configuration.

# NOSTR relay (must match customer's relay)
NOSTR_RELAY=${NOSTR_RELAY}

# Storage backend
STORAGE_URL=${STORAGE_URL}
STORAGE_AUTH_TOKEN=${STORAGE_AUTH_TOKEN}
STORAGE_PORT=8000

# DVM keypairs (hex secret keys — keep secret)
DVM_0_NSEC_HEX=${NSEC_KEYS[0]}
DVM_1_NSEC_HEX=${NSEC_KEYS[1]:-$(generate_hex "$PYTHON")}
DVM_2_NSEC_HEX=${NSEC_KEYS[2]:-$(generate_hex "$PYTHON")}
DVM_3_NSEC_HEX=${NSEC_KEYS[3]:-$(generate_hex "$PYTHON")}

# Logging
RUST_LOG=info
EOF

print_ok ".env written."

# Ensure .gitignore
if [ -f ".gitignore" ]; then
    grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
    grep -q "^!\.env\.example$" .gitignore || echo "!.env.example" >> .gitignore
else
    printf ".env\n!.env.example\n" > .gitignore
fi
print_ok ".env added to .gitignore"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║              DVM Setup Complete!                 ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Start your DVM(s):${NC}"
echo ""

# Build start commands based on num DVMs and storage
if [ "$NUM_DVMS" = "1" ]; then
    if [ -n "$STORAGE_PROFILE" ]; then
        echo -e "  ${GREEN}${BOLD}COMPOSE_PROFILES=local docker compose up dvm-0 storage --build${NC}"
    else
        echo -e "  ${GREEN}${BOLD}docker compose up dvm-0 --build${NC}"
    fi
elif [ "$NUM_DVMS" = "2" ]; then
    if [ -n "$STORAGE_PROFILE" ]; then
        echo -e "  ${GREEN}${BOLD}COMPOSE_PROFILES=local docker compose up dvm-0 dvm-1 storage --build${NC}"
    else
        echo -e "  ${GREEN}${BOLD}docker compose up dvm-0 dvm-1 --build${NC}"
    fi
elif [ "$NUM_DVMS" = "3" ]; then
    if [ -n "$STORAGE_PROFILE" ]; then
        echo -e "  ${GREEN}${BOLD}COMPOSE_PROFILES=local,dvm3 docker compose up dvm-0 dvm-1 dvm-2 storage --build${NC}"
    else
        echo -e "  ${GREEN}${BOLD}COMPOSE_PROFILES=dvm3 docker compose up dvm-0 dvm-1 dvm-2 --build${NC}"
    fi
else
    if [ -n "$STORAGE_PROFILE" ]; then
        echo -e "  ${GREEN}${BOLD}COMPOSE_PROFILES=local,dvm4 docker compose up --build${NC}"
    else
        echo -e "  ${GREEN}${BOLD}COMPOSE_PROFILES=dvm4 docker compose up dvm-0 dvm-1 dvm-2 dvm-3 --build${NC}"
    fi
fi

echo ""
echo "  DVMs will:"
echo "    1. Announce themselves on the relay"
echo "    2. Wait for job requests from any customer"
echo "    3. Execute training jobs (config comes from customer)"
echo "    4. Upload results to storage"
echo ""
echo "  Share with your customer:"
echo -e "  ${YELLOW}Relay: ${NOSTR_RELAY}${NC}"
echo ""
echo -e "  ${BOLD}Good luck! 🚀${NC}"
echo ""
