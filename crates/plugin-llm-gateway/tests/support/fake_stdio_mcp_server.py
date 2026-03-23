import json
import os
import sys


LOG_PATH = os.environ.get("TRP_STDIO_MCP_LOG")
REMOTE_TOOL = os.environ.get("TRP_STDIO_MCP_REMOTE_TOOL", "search_docs")
PROTOCOL_VERSION = os.environ.get("TRP_STDIO_MCP_PROTOCOL", "2025-11-25")
RESULT_PREFIX = os.environ.get("TRP_STDIO_MCP_RESULT_PREFIX", "Remote MCP result:")


def log_method(method: str) -> None:
    if not LOG_PATH:
        return
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(f"{method}\n")


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


for raw in sys.stdin:
    message = raw.strip()
    if not message:
        continue
    payload = json.loads(message)
    method = payload.get("method")
    log_method(method or "unknown")

    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "serverInfo": {
                        "name": "fake-stdio-mcp",
                        "version": "1.0.0",
                    },
                },
            }
        )
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": REMOTE_TOOL,
                            "description": "Search remote docs over stdio",
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        query = (
            payload.get("params", {})
            .get("arguments", {})
            .get("query", "")
        )
        send(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"{RESULT_PREFIX} {query}".strip(),
                        }
                    ]
                },
            }
        )
    else:
        send(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"unsupported method {method}",
                },
            }
        )
