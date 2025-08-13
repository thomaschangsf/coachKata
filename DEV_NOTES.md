# Commands
```bash
# Protocol Buffer workflow
uv run poe generate-code-from-protos

# Run the inference service
uv run poe run-inference-server

# Development tools
uv sync --dev
uv run poe check # runs all of the below
uv run ruff check --fix .
uv run pyright .
uv run pytest
uv run deptry .
```


