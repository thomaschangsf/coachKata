#!/usr/bin/env python3
"""
Main entry point for the Inference Director service.
"""

import uvicorn
from server.web_server import app


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
