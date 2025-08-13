#!/usr/bin/env python3
"""
FastAPI web server that wraps the gRPC InferenceDirector service.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .grpc_server import InferenceDirectorServer


# Pydantic models for HTTP API
class HealthStatusResponse(BaseModel):
    """Health status response model."""
    overall_status: str


class HealthStatusRequest(BaseModel):
    """Health status request model (empty)."""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the FastAPI application."""
    # Startup
    app.state.grpc_server = InferenceDirectorServer()
    await app.state.grpc_server.start()
    logging.info("FastAPI application started with gRPC server")

    yield

    # Shutdown
    await app.state.grpc_server.stop()
    logging.info("FastAPI application stopped")


# Create FastAPI app
app = FastAPI(
    title="Inference Director API",
    description="HTTP API wrapper for the InferenceDirector gRPC service",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthStatusResponse)
async def get_health_status() -> HealthStatusResponse:
    """
    Get the health status of the inference director.

    Returns:
        HealthStatusResponse: The current health status
    """
    try:
        # Get the gRPC servicer
        servicer = app.state.grpc_server.get_servicer()
        if not servicer:
            raise HTTPException(status_code=503, detail="gRPC server not available")

        # Create a mock request and context for the gRPC call
        from ..generated import inference_director_pb2  # type: ignore

        # Call the gRPC method directly
        request = inference_director_pb2.GetHealthStatusRequest()  # type: ignore
        response = servicer.GetHealthStatus(request, None)  # context is None for direct calls

        # Convert enum to string
        status_map = {
            0: "UNSPECIFIED",
            1: "HEALTHY",
            2: "UNHEALTHY"
        }
        status_str = status_map.get(response.overall_status, "UNKNOWN")

        return HealthStatusResponse(overall_status=status_str)

    except Exception as e:
        logging.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/health", response_model=HealthStatusResponse)
async def set_health_status(request: HealthStatusRequest) -> HealthStatusResponse:
    """
    Set the health status of the inference director.

    Args:
        request: Health status request (currently unused)

    Returns:
        HealthStatusResponse: The updated health status
    """
    try:
        # Get the gRPC servicer
        servicer = app.state.grpc_server.get_servicer()
        if not servicer:
            raise HTTPException(status_code=503, detail="gRPC server not available")

        # For now, just return the current status
        # In a real implementation, you might want to add a way to set the status
        from ..generated import inference_director_pb2  # type: ignore

        request_pb = inference_director_pb2.GetHealthStatusRequest()  # type: ignore
        response = servicer.GetHealthStatus(request_pb, None)

        status_map = {
            0: "UNSPECIFIED",
            1: "HEALTHY",
            2: "UNHEALTHY"
        }
        status_str = status_map.get(response.overall_status, "UNKNOWN")

        return HealthStatusResponse(overall_status=status_str)

    except Exception as e:
        logging.error(f"Error setting health status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Inference Director API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


def main():
    """Main function to run the FastAPI server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the server
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
