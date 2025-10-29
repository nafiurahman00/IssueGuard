"""
Main entry point for the FastAPI server.

Usage:
    python main.py
    python main.py --host 0.0.0.0 --port 8080
    python main.py --reload  # Development mode
"""

import argparse
import uvicorn

from api.app import create_app
from api.config import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FastAPI server for secret detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.HOST,
        help=f"Host to bind to (default: {settings.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.PORT,
        help=f"Port to bind to (default: {settings.PORT})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
