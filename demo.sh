#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY="$DIR/target/release/tiny-reverse-proxy-rust"
CONFIG="$DIR/bench/demo.toml"
CERT="$DIR/.tls/cert.pem"

cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$UPSTREAM_PID" 2>/dev/null || true
    kill "$PROXY_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

# 1. Start a simple upstream on port 9000
python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type','text/plain')
        self.end_headers()
        self.wfile.write(b'Hello from upstream!')
    def log_message(self, *a): pass
HTTPServer(('127.0.0.1', 9000), H).serve_forever()
" &
UPSTREAM_PID=$!
sleep 0.3

# 2. Start the proxy
"$PROXY" "$CONFIG" &
PROXY_PID=$!
sleep 1

# 3. Trust the self-signed cert (macOS)
if [ -f "$CERT" ]; then
    echo ""
    echo "==> Trusting self-signed certificate (requires sudo)..."
    sudo security add-trusted-cert -d -r trustRoot \
        -k /Library/Keychains/System.keychain "$CERT"
    echo "==> Certificate trusted! Open https://localhost:8443 in your browser."
else
    echo "==> Warning: cert not found at $CERT"
fi

echo ""
echo "==> Proxy running on https://localhost:8443"
echo "==> Metrics at http://localhost:9090/metrics"
echo "==> Press Ctrl+C to stop"
echo ""

wait
