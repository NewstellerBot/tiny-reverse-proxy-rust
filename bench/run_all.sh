#!/bin/bash
# Benchmark all proxies against the same upstream.
# Usage: bash bench/run_all.sh
#
# Uses hey's default keep-alive behavior and runs each proxy
# independently to avoid ephemeral port exhaustion cascades.

set -uo pipefail

DURATION="10s"
CONCURRENCY=50

cleanup_all() {
    pkill -f "upstream-fast" 2>/dev/null || true
    pkill -f "tiny-reverse-proxy-rust bench" 2>/dev/null || true
    pkill -f "caddy run" 2>/dev/null || true
    pkill -f "pingora-bench-proxy" 2>/dev/null || true
    sleep 1
}

wait_for_port() {
    local port=$1 tries=20
    while ! curl -sf "http://127.0.0.1:$port/" > /dev/null 2>&1; do
        tries=$((tries - 1))
        if [ $tries -le 0 ]; then echo "FAIL: port $port never came up"; exit 1; fi
        sleep 0.25
    done
}

bench() {
    local name=$1 port=$2
    printf "%-25s" "$name"
    # Warmup
    hey -z 2s -c "$CONCURRENCY" "http://127.0.0.1:$port/" > /dev/null 2>&1
    sleep 1
    # Actual benchmark
    local result
    result=$(hey -z "$DURATION" -c "$CONCURRENCY" "http://127.0.0.1:$port/" 2>&1)
    local rps=$(echo "$result" | grep "Requests/sec" | awk '{print $2}')
    local avg=$(echo "$result" | grep "Average:" | head -1 | awk '{print $2}')
    local p50=$(echo "$result" | grep "50%" | awk '{print $3}')
    local p99=$(echo "$result" | grep "99%" | awk '{print $3}')
    local errors=$(echo "$result" | grep -c "Error distribution" || true)
    local err_count=$(echo "$result" | grep -E "^\s+\[" | grep -v "200" | awk '{sum += $1} END {print sum+0}')
    printf "%10s req/s   avg=%ss   p50=%ss   p99=%ss   errors=%s\n" "$rps" "$avg" "$p50" "$p99" "$err_count"
}

# Clean slate
cleanup_all

# Start upstream (shared across all benchmarks)
./target/release/upstream-fast &
UPSTREAM_PID=$!
wait_for_port 9000

echo "================================================================="
echo "  Proxy Benchmark  |  c=$CONCURRENCY  |  duration=$DURATION"
echo "================================================================="
echo ""

# --- Direct upstream baseline ---
bench "Direct upstream" 9000
sleep 3

# --- tiny-reverse-proxy ---
RUST_LOG=error ./target/release/tiny-reverse-proxy-rust bench/proxy.toml &
wait_for_port 8100
bench "tiny-reverse-proxy" 8100
pkill -f "tiny-reverse-proxy-rust bench" 2>/dev/null || true
sleep 3

# --- Caddy ---
caddy run --config bench/Caddyfile --adapter caddyfile > /dev/null 2>&1 &
wait_for_port 8200
bench "Caddy" 8200
pkill -f "caddy run" 2>/dev/null || true
sleep 3

# --- Pingora ---
./bench/pingora-proxy/target/release/pingora-bench-proxy > /dev/null 2>&1 &
wait_for_port 8300
bench "Pingora" 8300
pkill -f "pingora-bench-proxy" 2>/dev/null || true

echo ""
echo "Done."

# Final cleanup
kill $UPSTREAM_PID 2>/dev/null || true
