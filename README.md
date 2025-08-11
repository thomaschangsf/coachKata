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

Compile Protocol Buffers:
```bash
uv run betterproto2-compile
```

### Task Runner

Run tasks with poethepoet:
```bash
uv run poe <task-name>
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
- **betterproto2-compile**: Protocol Buffer compilation
- **grpcio-tools**: gRPC code generation
- **httpx**: Modern HTTP client for testing
- **poethepoet**: Task runner for automation

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
3. Run tests: `uv run pytest`
4. Format and lint code: `uv run ruff check --fix . && uv run ruff format .`
5. Run type checking: `uv run pyright .`
6. Check dependencies: `uv run deptry .`
7. Submit a pull request

## License

[Add your license information here]
