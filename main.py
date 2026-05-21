"""Entry point for the OpenNPC API Runtime service."""

import argparse
import sys
import uvicorn
from opennpc.api.service import require_api_deps


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the OpenNPC runtime server."""
    parser = argparse.ArgumentParser(description="OpenNPC Runtime API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port to bind the server to (default: 8787)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )
    return parser.parse_args()


def main() -> None:
    """Verify dependencies and start the Uvicorn server."""
    try:
        require_api_deps()
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    print(f"Starting OpenNPC API Server on http://{args.host}:{args.port}")
    if args.reload:
        print("Auto-reload enabled.")

    uvicorn.run(
        "opennpc.api.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
