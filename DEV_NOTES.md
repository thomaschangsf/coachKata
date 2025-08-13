# Commands
```bash
# Protocol Buffer workflow
uv run poe generate-code-from-protos

# Run the inference service
uv run poe run-inference-server
    cd services/inference && uv run python app/main.py
    # python path = where python interpretor looks for modules
    # absolute path: is relative to the python path 
    # python path comes from 
    #     (1) current working dir, bc cd services/inference 
    #     (2) where the script main.py (__main__) runs from, app/main.py
    #     (3) -m flag
    # SIDE: python module vs package
    #     module=python file, has __name__ attr; 
    #     package=diretory, has __init__.py
    # SIDE: Relative Path
    #     EX: from ..generated import
    #     is relative to the current module's location
    #     IMOW: it's bottom level up
lsof -i :8000
pkill -f "python app/main.py"

# Trigger
curl http://localhost:8000/health

cd services/inference/app
python client_test.py

# Development tools
uv sync --dev
uv run poe check # runs all of the below
uv run ruff check --fix .
uv run pyright .
uv run pytest
uv run deptry .
```


