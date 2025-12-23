#!/usr/bin/env python3
"""
Test script to demonstrate -m vs direct execution
"""

import sys

print("=== Module Execution Test ===")
print(f"Script name: {__name__}")
print("Python path:")
for path in sys.path:
    print(f"  {path}")
print(f"Current working directory: {sys.path[0]}")
