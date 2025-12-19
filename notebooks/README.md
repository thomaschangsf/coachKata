# Jupyter Notebooks

This directory contains Jupyter notebooks for exploring and experimenting with the Coach Kata project.

## Setup

1. **Install Jupyter dependencies:**
   ```bash
   uv add --dev jupyter notebook ipykernel matplotlib pandas numpy mediapipe
   ```

2. **Start Jupyter Notebook:**
   ```bash
   uv run poe jupyter
   ```
   
   Or start JupyterLab:
   ```bash
   uv run poe jupyter-lab
   ```

3. **Alternative: Use VS Code**
   - Install the Jupyter extension in VS Code
   - Open any `.ipynb` file
   - Select the kernel when prompted

## Available Notebooks

- `coach_kata_examples.ipynb` - Basic examples showing how to work with the project

## Tips

- **Import your modules:** Add the project root to Python path in notebooks:
  ```python
  import sys
  import os
  sys.path.insert(0, os.path.abspath('..'))
  ```

- **Test the inference server:** Make sure the server is running:
  ```bash
  uv run poe run-inference-server
  ```

- **Git integration:** Notebooks are tracked in git, but outputs are cleared to avoid conflicts

## Environment

The notebooks use the same Python environment as your project, so you have access to:
- All project dependencies
- Your custom modules
- Generated protobuf code
- Inference server client code 