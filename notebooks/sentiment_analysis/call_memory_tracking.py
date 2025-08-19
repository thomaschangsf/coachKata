#!/usr/bin/env python3
"""
How to call onnx_memory_tracking.py in different ways

This script shows various methods to use the memory tracking functionality:
1. Import and call functions directly
2. Run as a module
3. Jupyter notebook cell code
"""

import sys
import os
from pathlib import Path

# Add the notebooks directory to the path
notebooks_dir = Path(__file__).parent
sys.path.insert(0, str(notebooks_dir))

# Method 1: Import and use specific functions
def method1_import_functions():
    """Import specific functions and use them directly."""
    print("=== Method 1: Import Specific Functions ===")
    
    try:
        from onnx_memory_tracking import track_with_psutil, get_onnx_model_info
        
        # Run just the primary tracking
        track_with_psutil()
        
        # Get model information
        get_onnx_model_info()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure onnx_memory_tracking.py is in the same directory")

# Method 2: Import the entire module
def method2_import_module():
    """Import the entire module and control execution."""
    print("\n=== Method 2: Import Entire Module ===")
    
    try:
        import onnx_memory_tracking as omt
        
        # You can now access any function from the module
        print("Available functions:")
        print([func for func in dir(omt) if not func.startswith('_')])
        
        # Run specific tracking methods
        omt.track_with_psutil()
        
    except ImportError as e:
        print(f"Import error: {e}")

# Method 3: Run as a subprocess
def method3_subprocess():
    """Run the script as a subprocess."""
    print("\n=== Method 3: Run as Subprocess ===")
    
    import subprocess
    
    script_path = notebooks_dir / "onnx_memory_tracking.py"
    
    if script_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=notebooks_dir
            )
            
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
                
        except Exception as e:
            print(f"Error running subprocess: {e}")
    else:
        print(f"Script not found at {script_path}")

# Method 4: Execute the script directly
def method4_execute_script():
    """Execute the script directly using exec()."""
    print("\n=== Method 4: Execute Script Directly ===")
    
    script_path = notebooks_dir / "onnx_memory_tracking.py"
    
    if script_path.exists():
        try:
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            # Execute the script in the current namespace
            exec(script_content)
            
        except Exception as e:
            print(f"Error executing script: {e}")
    else:
        print(f"Script not found at {script_path}")

if __name__ == "__main__":
    print("Different ways to call onnx_memory_tracking.py")
    print("=" * 50)
    
    method1_import_functions()
    method2_import_module()
    method3_subprocess()
    method4_execute_script() 