"""Minimal HTTP server as an upstream target for benchmarking."""
import http.server
import sys

class Handler(http.server.BaseHTTPRequestHandler):
    response_body = b"Hello from upstream!\n"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format, *args):
        pass  # silence logs

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"Upstream listening on 127.0.0.1:{port}")
    server.serve_forever()
