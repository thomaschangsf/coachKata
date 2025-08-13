#!/usr/bin/env python3
"""
gRPC server implementation for the InferenceDirector service.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import grpc
from grpc import aio

# Import the generated protobuf code
from ..generated import (
    inference_director_pb2,  # type: ignore
    inference_director_pb2_grpc,  # type: ignore
)


class InferenceDirectorServicer(inference_director_pb2_grpc.InferenceDirectorServicer):  # type: ignore
    """Implementation of the InferenceDirector service."""

    def __init__(self):
        """Initialize the servicer."""
        self.logger = logging.getLogger(__name__)
        self._health_status = inference_director_pb2.HealthStatus.HEALTH_STATUS_HEALTHY  # type: ignore

    def GetHealthStatus(  # type: ignore
        self,
        request: inference_director_pb2.GetHealthStatusRequest,  # type: ignore
        context: grpc.aio.ServicerContext  # type: ignore
    ) -> inference_director_pb2.HealthStatusResponse:  # type: ignore
        """
        Get the health status of the inference director.

        Args:
            request: The health status request (empty)
            context: gRPC context

        Returns:
            HealthStatusResponse with the current health status
        """
        self.logger.info("Health status requested")

        response = inference_director_pb2.HealthStatusResponse(  # type: ignore
            overall_status=self._health_status  # type: ignore
        )

        return response  # type: ignore

    def set_health_status(self, status: inference_director_pb2.HealthStatus) -> None:  # type: ignore
        """
        Set the health status of the service.

        Args:
            status: The new health status
        """
        self.logger.info(f"Setting health status to: {status}")
        self._health_status = status  # type: ignore


class InferenceDirectorServer:
    """gRPC server for the InferenceDirector service."""

    def __init__(self, host: str = "0.0.0.0", port: int = 50051):
        """
        Initialize the gRPC server.

        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self._server: aio.Server | None = None
        self._servicer: InferenceDirectorServicer | None = None

    async def start(self) -> None:
        """Start the gRPC server."""
        self.logger.info(f"Starting gRPC server on {self.host}:{self.port}")

        # Create the server
        self._server = aio.server(
            ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000),
            ]
        )

        # Create and add the servicer
        self._servicer = InferenceDirectorServicer()
        inference_director_pb2_grpc.add_InferenceDirectorServicer_to_server(  # type: ignore
            self._servicer, self._server
        )

        # Bind the server
        listen_addr = f"{self.host}:{self.port}"
        self._server.add_insecure_port(listen_addr)

        # Start the server
        await self._server.start()
        self.logger.info(f"gRPC server started successfully on {listen_addr}")

    async def stop(self) -> None:
        """Stop the gRPC server."""
        if self._server:
            self.logger.info("Stopping gRPC server...")
            await self._server.stop(grace=5)
            self.logger.info("gRPC server stopped")

    async def wait_for_termination(self) -> None:
        """Wait for the server to terminate."""
        if self._server:
            await self._server.wait_for_termination()

    def get_servicer(self) -> InferenceDirectorServicer | None:
        """Get the servicer instance."""
        return self._servicer


async def main():
    """Main function to run the gRPC server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and start the server
    server = InferenceDirectorServer()

    try:
        await server.start()
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
