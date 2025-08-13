#!/usr/bin/env python3
"""
Test client for the InferenceDirector gRPC service.
"""

import asyncio
import logging

import grpc
from generated import (
    inference_director_pb2,  # type: ignore
    inference_director_pb2_grpc,  # type: ignore
)


async def test_grpc_client():
    """Test the gRPC client connection."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a channel
    channel = grpc.aio.insecure_channel('localhost:50051')

    # Create a stub
    stub = inference_director_pb2_grpc.InferenceDirectorStub(channel)  # type: ignore

    try:
        # Create a request
        request = inference_director_pb2.GetHealthStatusRequest()  # type: ignore

        # Make the call
        logger.info("Making gRPC call to GetHealthStatus...")
        response = await stub.GetHealthStatus(request)  # type: ignore

        # Print the response
        status_map = {
            0: "UNSPECIFIED",
            1: "HEALTHY",
            2: "UNHEALTHY"
        }
        status_str = status_map.get(response.overall_status, "UNKNOWN")  # type: ignore
        logger.info(f"Health status: {status_str}")

    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await channel.close()


async def test_http_client():
    """Test the HTTP client connection."""
    import httpx

    logger = logging.getLogger(__name__)

    async with httpx.AsyncClient() as client:
        try:
            # Test the health endpoint
            logger.info("Making HTTP call to /health...")
            response = await client.get("http://localhost:8000/health")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health status: {data['overall_status']}")
            else:
                logger.error(f"HTTP error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error: {e}")


async def main():
    """Run both tests."""
    logger = logging.getLogger(__name__)

    logger.info("Testing gRPC client...")
    await test_grpc_client()

    logger.info("Testing HTTP client...")
    await test_http_client()


if __name__ == "__main__":
    asyncio.run(main())
