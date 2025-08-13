# Inference Director Service Architecture

## Overview

The Inference Director service is built with a layered architecture that provides both HTTP and gRPC APIs. Here's how the three main files work together:

## File Structure

```
services/inference/app/
├── main.py                    # Entry point
├── server/
│   ├── web_server.py         # FastAPI HTTP layer
│   └── grpc_server.py        # gRPC implementation
└── generated/                 # Protocol Buffer code
    ├── inference_director_pb2.py
    └── inference_director_pb2_grpc.py
```

## Detailed Architecture

### 1. main.py - Entry Point
```python
import uvicorn
from server.web_server import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Purpose:**
- Simple launcher script
- Starts the FastAPI application using uvicorn
- No business logic, just a convenient entry point

### 2. web_server.py - HTTP API Layer
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create and start gRPC server
    app.state.grpc_server = InferenceDirectorServer()
    await app.state.grpc_server.start()
    
    yield
    
    # Shutdown: Stop gRPC server
    await app.state.grpc_server.stop()

@app.get("/health")
async def get_health_status():
    # Call gRPC method directly (not over network)
    servicer = app.state.grpc_server.get_servicer()
    response = servicer.GetHealthStatus(request, None)
    return HealthStatusResponse(overall_status=status_str)
```

**Purpose:**
- Creates FastAPI application with HTTP endpoints
- **Manages gRPC server lifecycle** via `lifespan()` function
- **HTTP endpoints call gRPC methods directly** (in-process, not over network)
- Provides REST API wrapper around gRPC functionality
- Handles HTTP request/response formatting

### 3. grpc_server.py - gRPC Implementation
```python
class InferenceDirectorServicer(inference_director_pb2_grpc.InferenceDirectorServicer):
    def GetHealthStatus(self, request, context):
        return inference_director_pb2.HealthStatusResponse(
            overall_status=self._health_status
        )

class InferenceDirectorServer:
    async def start(self):
        # Create async gRPC server
        self._server = aio.server(ThreadPoolExecutor(max_workers=10))
        # Add servicer
        inference_director_pb2_grpc.add_InferenceDirectorServicer_to_server(
            self._servicer, self._server
        )
        # Bind to port 50051
        self._server.add_insecure_port(f"{self.host}:{self.port}")
        await self._server.start()
```

**Purpose:**
- Contains the actual business logic
- Implements `InferenceDirectorServicer` with `GetHealthStatus()`
- Runs async gRPC server on port 50051
- Manages health status state
- Handles gRPC protocol details

## Data Flow

### HTTP Request Flow
```
HTTP Client (port 8000)
    ↓
main.py → uvicorn
    ↓
web_server.py → FastAPI app
    ↓
HTTP endpoint (e.g., /health)
    ↓
Calls gRPC method directly (in-process)
    ↓
grpc_server.py → InferenceDirectorServicer
    ↓
Returns response
    ↓
web_server.py → Formats as HTTP response
    ↓
HTTP Client
```

### gRPC Request Flow
```
gRPC Client (port 50051)
    ↓
grpc_server.py → InferenceDirectorServer
    ↓
InferenceDirectorServicer
    ↓
Returns gRPC response
    ↓
gRPC Client
```

## Key Points

1. **Single Process**: All three components run in the same process
2. **Dual Protocol**: Supports both HTTP (8000) and gRPC (50051)
3. **Direct Calls**: HTTP endpoints call gRPC methods directly, not over network
4. **Lifecycle Management**: FastAPI manages gRPC server startup/shutdown
5. **State Sharing**: gRPC server state is accessible to HTTP endpoints

## Why This Architecture?

1. **Flexibility**: Clients can choose HTTP or gRPC
2. **Performance**: Direct method calls (no network overhead for HTTP→gRPC)
3. **Simplicity**: Single process, easy to deploy
4. **Documentation**: FastAPI provides automatic API docs
5. **Extensibility**: Easy to add new endpoints and methods

## Adding New Features

1. **New gRPC Method**: Add to `grpc_server.py`
2. **New HTTP Endpoint**: Add to `web_server.py`
3. **New Proto Definition**: Update proto file and regenerate
4. **Configuration**: Modify `main.py` or add environment variables 