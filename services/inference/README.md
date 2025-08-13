# Inference Director Service

A FastAPI web service that wraps a gRPC InferenceDirector service, providing both HTTP and gRPC endpoints.

## Features

- **gRPC Server**: Implements the InferenceDirector service with health status monitoring
- **HTTP API**: FastAPI wrapper providing REST endpoints
- **Protocol Buffer Integration**: Uses generated protobuf code from `.proto` definitions
- **MediaPipe Support**: Ready for media processing tasks

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HTTP Client   │    │   gRPC Client   │    │   FastAPI App   │
│   (Port 8000)   │    │   (Port 50051)  │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   InferenceDirector       │
                    │   gRPC Service            │
                    │   (Health Status)         │
                    └───────────────────────────┘
```

## Quick Start

### 1. Generate Protocol Buffer Code

First, generate the Python code from your `.proto` files:

```bash
uv run poe generate-code-from-protos
```

### 2. Run the Server

Start the FastAPI server (which includes the gRPC server):

```bash
uv run poe run-inference-server
```

Or run directly:

```bash
cd services/inference
uv run python app/main.py
```

### 3. Test the Service

The server will be available at:
- **HTTP API**: http://localhost:8000
- **gRPC Server**: localhost:50051
- **API Documentation**: http://localhost:8000/docs

## API Endpoints

### HTTP Endpoints

#### GET /health
Get the health status of the inference director.

**Response:**
```json
{
  "overall_status": "HEALTHY"
}
```

#### POST /health
Set the health status (currently returns current status).

**Response:**
```json
{
  "overall_status": "HEALTHY"
}
```

#### GET /
Root endpoint with API information.

**Response:**
```json
{
  "message": "Inference Director API",
  "version": "0.1.0",
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "openapi": "/openapi.json"
  }
}
```

### gRPC Endpoints

#### GetHealthStatus
Get the health status of the inference director.

**Request:** `GetHealthStatusRequest` (empty)
**Response:** `HealthStatusResponse` with `overall_status` field

## Health Status Values

The service supports three health status values:
- `UNSPECIFIED` (0): Default/unspecified status
- `HEALTHY` (1): Service is healthy and functioning normally
- `UNHEALTHY` (2): Service is unhealthy or experiencing issues

## Development

### Project Structure

```
services/inference/
├── app/
│   ├── generated/           # Generated protobuf code
│   │   ├── inference_director_pb2.py
│   │   └── inference_director_pb2_grpc.py
│   ├── server/
│   │   ├── grpc_server.py   # gRPC server implementation
│   │   └── web_server.py    # FastAPI web server
│   ├── main.py              # Main entry point
│   └── test_client.py       # Test client
├── pyproject.toml           # Package configuration
└── README.md               # This file
```

### Testing

Run the test client to verify both HTTP and gRPC endpoints:

```bash
cd services/inference
uv run python app/test_client.py
```

### Adding New gRPC Methods

1. **Update the proto file** in `protos/inference_director.proto`
2. **Regenerate the code**: `uv run poe generate-code-from-protos`
3. **Implement the method** in `app/server/grpc_server.py`
4. **Add HTTP wrapper** in `app/server/web_server.py` (if needed)

## Dependencies

- **FastAPI**: Web framework
- **uvicorn**: ASGI server
- **grpcio**: gRPC implementation
- **pydantic**: Data validation
- **mediapipe**: Media processing (ready for future use)

## Configuration

The service can be configured via environment variables:
- `HOST`: Server host (default: 0.0.0.0)
- `HTTP_PORT`: HTTP port (default: 8000)
- `GRPC_PORT`: gRPC port (default: 50051)

