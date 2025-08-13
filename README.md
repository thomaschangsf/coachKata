# Coach Kata

A Python monorepo for ML coaching applications using `uv` as the package manager.

## Project Structure

```
coachKata/
├── libraries/     # Shared libraries and utilities
├── models/        # ML models and training code
├── protos/        # Protocol buffers and data schemas
├── scripts/       # Utility scripts and tools
├── services/      # Service packages
│   └── inference/ # Model inference and serving
├── app/           # Main application code
└── pyproject.toml # Workspace configuration
```

## Prerequisites

- Python 3.11 or higher
- `uv` package manager

## Setup Instructions

### 1. Install uv

If you don't have `uv` installed, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then add it to your PATH:
```bash
source $HOME/.local/bin/env
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd coachKata
```

### 3. Install Dependencies

Install all dependencies for the workspace:

```bash
uv sync
```

This will install dependencies for all packages in the monorepo.

### 4. Install Development Dependencies

Install development tools (pytest, ruff, pyright, etc.):

```bash
uv sync --dev
```

## Development Workflow

### Running Tests

Run tests across all packages:
```bash
uv run pytest
```

Run tests for a specific package:
```bash
uv run pytest libraries/
```

Run tests with coverage:
```bash
uv run pytest --cov
```

### Code Formatting and Linting

Format and lint all code with ruff:
```bash
uv run ruff check --fix .
```

Format code only:
```bash
uv run ruff format .
```

Sort imports:
```bash
uv run ruff check --select I --fix .
```

### Type Checking

Run pyright type checking:
```bash
uv run pyright .
```

### Dependency Management

Check for unused dependencies:
```bash
uv run deptry .
```

### Protocol Buffer Compilation

Generate Python code from Protocol Buffer definitions:
```bash
uv run poe generate-code-from-protos
```

This command:
- Reads configuration from `protos/config/proto-config.json`
- Finds all `.proto` files in the `protos/` directory
- Generates Python code using `grpcio-tools.protoc`
- Outputs generated files to `services/inference/app/generated/`

### Task Runner

Run tasks with poethepoet:
```bash
uv run poe <task-name>
```

#### Available Tasks

**Code Quality Check (All-in-one):**
```bash
uv run poe check
```

**Protocol Buffer Generation:**
```bash
uv run poe generate-code-from-protos
```

**Inference Service:**
```bash
uv run poe run-inference-server
```

### Adding Dependencies

Add a dependency to a specific package:
```bash
cd libraries
uv add package-name
```

Add a development dependency:
```bash
uv add --dev package-name
```

### Running Scripts

Run a script in a specific package:
```bash
uv run python scripts/your_script.py
```

## Development Tools

The monorepo includes the following development tools:

- **pytest**: Testing framework with async support and coverage reporting
- **ruff**: Fast Python linter and formatter (replaces black, isort, flake8)
- **pyright**: Type checker by Microsoft (replaces mypy)
- **deptry**: Dependency management and analysis
- **grpcio-tools**: Protocol Buffer and gRPC code generation
- **betterproto**: Modern Protocol Buffer library with dataclass support
- **httpx**: Modern HTTP client for testing
- **poethepoet**: Task runner for automation

## Services

### Inference Director Service

The inference service provides both gRPC and HTTP APIs for model inference management.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    main.py (Entry Point)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              uvicorn.run(web_server.app)            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                web_server.py (FastAPI App)                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FastAPI App with HTTP endpoints (/health, /docs)  │   │
│  │                                                     │   │
│  │  lifespan() function:                               │   │
│  │  ├─ startup: creates grpc_server                    │   │
│  │  └─ shutdown: stops grpc_server                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              grpc_server.py (gRPC Implementation)           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  InferenceDirectorServicer                         │   │
│  │  ├─ GetHealthStatus() method                       │   │
│  │  └─ Health status management                       │   │
│  │                                                     │   │
│  │  InferenceDirectorServer                           │   │
│  │  ├─ Async gRPC server (port 50051)                 │   │
│  │  └─ Lifecycle management                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### File Relationships

**1. main.py** - Entry Point
- Simple launcher that starts the FastAPI app
- Runs uvicorn server on port 8000
- Imports and runs `web_server.app`

**2. web_server.py** - HTTP API Layer
- Creates FastAPI application with HTTP endpoints
- **Manages gRPC server lifecycle** via `lifespan()` function
- **HTTP endpoints call gRPC methods directly** (not over network)
- Provides REST API wrapper around gRPC functionality

**3. grpc_server.py** - gRPC Implementation
- Contains the actual business logic
- Implements `InferenceDirectorServicer` with `GetHealthStatus()`
- Runs async gRPC server on port 50051
- Manages health status state

#### How They Work Together

1. **main.py** starts the application
2. **web_server.py** creates FastAPI app and starts gRPC server
3. **HTTP requests** → **web_server.py** → **calls gRPC methods directly** → **grpc_server.py**
4. **gRPC requests** → **grpc_server.py** directly (port 50051)

**Quick Start:**
```bash
# Generate Protocol Buffer code
uv run poe generate-code-from-protos

# Run the inference server
uv run poe run-inference-server
```

**Service Endpoints:**
- **HTTP API**: http://localhost:8000 (with docs at /docs)
- **gRPC Server**: localhost:50051

**Features:**
- Health status monitoring
- Protocol Buffer integration
- MediaPipe support
- FastAPI web interface

For detailed documentation, see [services/inference/README.md](services/inference/README.md).

## Package Development

Each directory (`libraries`, `models`, `protos`, `scripts`, `services/*`, `app`) can contain its own Python package with a `pyproject.toml` file.

### Creating a New Package

1. Create a new directory in the appropriate location
2. Add a `pyproject.toml` file
3. The package will automatically be included in the workspace

### Package Dependencies

Packages can depend on each other by adding the package name to their dependencies:

```toml
[project]
dependencies = [
    "coachkata-libraries",
    "other-package-name",
]
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run all checks: `uv run poe check`
4. Submit a pull request

**Note:** The `check` task runs:
- `ruff check --fix .` (linting and auto-fixing)
- `pyright .` (type checking)
- `pytest` (testing)
- `deptry .` (dependency analysis)

## License

[Add your license information here]
