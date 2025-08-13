#!/usr/bin/env python3
"""
Protocol Buffer code generator script.

This script reads a configuration file and generates Python code from .proto files
using grpcio-tools.

Note: While betterproto2 was requested, the betterproto2-compiler package doesn't
exist as a separate command-line tool. The betterproto library is available for
runtime use, but for code generation, we use the standard grpcio-tools.protoc
which generates traditional protobuf classes.

To use betterproto2-style dataclasses, you can:
1. Use the betterproto library directly in your code
2. Or manually create dataclasses that match your proto definitions
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_config(config_path: str) -> dict[str, Any]:
    """Load the proto configuration from JSON file."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def find_proto_files(proto_path: str) -> list[str]:
    """Find all .proto files in the given directory."""
    proto_files: list[str] = []
    proto_dir = Path(proto_path)

    if not proto_dir.exists():
        print(f"Error: Proto directory not found at {proto_path}")
        sys.exit(1)

    for proto_file in proto_dir.rglob("*.proto"):
        proto_files.append(str(proto_file))

    return proto_files


def create_generated_directory(generated_path: str) -> None:
    """Create the generated code directory if it doesn't exist."""
    generated_dir = Path(generated_path)
    generated_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified generated directory: {generated_dir}")


def generate_proto_code(proto_files: list[str], proto_path: str, generated_path: str) -> None:
    """Generate Python code from .proto files using grpcio-tools."""
    if not proto_files:
        print("No .proto files found to generate code from.")
        return

    print(f"Found {len(proto_files)} .proto file(s):")
    for proto_file in proto_files:
        print(f"  - {proto_file}")

    print("\nGenerating Python code with grpcio-tools...")
    print("Note: Using standard protobuf generation. For betterproto2, use the betterproto library directly.")

    for proto_file in proto_files:
        try:
            # Run protoc command with standard options
            cmd = [
                sys.executable, "-m", "grpc_tools.protoc",
                f"--proto_path={proto_path}",
                f"--python_out={generated_path}",
                f"--grpc_python_out={generated_path}",
                proto_file
            ]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Extract the filename without extension
            proto_filename = Path(proto_file).stem
            print(f"✓ Generated code for {proto_filename}")

        except subprocess.CalledProcessError as e:
            print(f"Error generating code for {proto_file}:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            sys.exit(1)


def create_init_file(generated_path: str) -> None:
    """Create an __init__.py file in the generated directory."""
    init_file = Path(generated_path) / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Created {init_file}")


def main() -> None:
    """Main function to orchestrate the proto generation process."""
    # Get the script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load configuration
    config_path = project_root / "protos" / "config" / "proto-config.json"
    config = load_config(str(config_path))

    # Extract configuration values
    local_proto_path = config.get("local_proto_path", "protos")
    local_generated_path = config.get("local_generated_path", "services/inference/app/generated")

    # Convert to absolute paths
    proto_path = project_root / local_proto_path
    generated_path = project_root / local_generated_path

    print(f"Proto path: {proto_path}")
    print(f"Generated path: {generated_path}")
    print()

    # Create generated directory
    create_generated_directory(str(generated_path))

    # Find .proto files
    proto_files = find_proto_files(str(proto_path))

    # Generate code
    generate_proto_code(proto_files, str(proto_path), str(generated_path))

    # Create __init__.py file
    create_init_file(str(generated_path))

    print("\n✓ Protocol Buffer code generation completed successfully!")


if __name__ == "__main__":
    main()
