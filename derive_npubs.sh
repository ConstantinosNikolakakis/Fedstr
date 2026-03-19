#!/usr/bin/env bash
# scripts/derive_npubs.sh
# Derives the npub for each DVM from its NSEC_HEX and prints the DVM_NPUBS line.
# Usage: source .env && bash scripts/derive_npubs.sh
#
# Requires: fedstr-dvm binary already built (docker build or cargo build)

set -euo pipefail

BINARY=${DVM_BIN:-./target/release/fedstr-dvm}

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run 'cargo build --release' or 'docker compose build' first."
    exit 1
fi

npubs=()
for i in 0 1 2 3; do
    varname="DVM_${i}_NSEC_HEX"
    nsec="${!varname:-}"
    if [ -z "$nsec" ]; then
        break
    fi
    npub=$("$BINARY" --keypair-hex "$nsec" --print-pubkey 2>/dev/null)
    npubs+=("$npub")
    echo "DVM $i: $npub"
done

joined=$(IFS=','; echo "${npubs[*]}")
echo ""
echo "Add this to your .env:"
echo "DVM_NPUBS=$joined"
echo "NUM_DVMS=${#npubs[@]}"
