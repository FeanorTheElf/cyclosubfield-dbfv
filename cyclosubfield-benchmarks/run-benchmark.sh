#!/usr/bin/env bash
# Runs the benchmarks with the Galois keys streamed off an SSD.
#
# At the large parameter sets the key material does not fit in RAM (tens of gigabytes at
# field_65536), so `DBFV_KEY_STORE_DIR` points the key store at a fast device and the keys are
# held in a memory-mapped file there; see `cyclosubfield_dbfv::keystore`.
#
# The cgroup limit is not optional. The key store's pages are clean, file-backed and therefore
# cheap for the kernel to keep, so without a limit the global reclaim balance (vm.swappiness
# defaults to 60) prefers to swap out *other* processes' anonymous pages over dropping them —
# which makes the rest of the machine crawl while the benchmark runs. Inside a cgroup the
# pressure is confined: when the benchmark reaches MemoryHigh the kernel reclaims its own page
# cache, which costs nothing but a re-read from the SSD, and leaves everything else alone.
#
# MemorySwapMax=0 keeps the benchmark itself out of swap: its resident working set (the
# transform's `babys` buffer) must stay in RAM, and swapping it would be catastrophic — the
# giant-step loop rereads that buffer once per giant step.

set -euo pipefail

# the script lives in the benchmark crate, the workspace (and hence `target/`) is one level up
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STORE_DIR="${DBFV_KEY_STORE_DIR:-/mnt/nvme_ssd/dbfv}"
# leave room for the desktop; the benchmark reclaims its own key cache beyond this
MEM_HIGH="${MEM_HIGH:-44G}"
MEM_MAX="${MEM_MAX:-52G}"

if ! mountpoint -q "$(df --output=target "$STORE_DIR" 2>/dev/null | tail -1)" 2>/dev/null; then
    echo "warning: $STORE_DIR does not look like a separate mount; make sure it is on the NVMe" >&2
fi
mkdir -p "$STORE_DIR"

avail=$(df --output=avail -BG "$STORE_DIR" | tail -1 | tr -dc '0-9')
if [ "${avail:-0}" -lt 100 ]; then
    echo "error: only ${avail}G free on $STORE_DIR; the key store needs ~100G of headroom" >&2
    exit 1
fi

echo "key store : $STORE_DIR (${avail}G free)"
echo "cgroup    : MemoryHigh=$MEM_HIGH MemoryMax=$MEM_MAX MemorySwapMax=0"

cargo build --release --manifest-path "$WORKSPACE_ROOT/Cargo.toml" -p cyclosubfield-benchmarks

exec systemd-run --user --scope --collect \
    -p MemoryHigh="$MEM_HIGH" \
    -p MemoryMax="$MEM_MAX" \
    -p MemorySwapMax=0 \
    -E DBFV_KEY_STORE_DIR="$STORE_DIR" \
    -- "$WORKSPACE_ROOT/target/release/cyclosubfield-benchmarks" "$@"
