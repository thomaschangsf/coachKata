# Why We Ignore Local Modules in Deptry

## What is Deptry?

**Deptry** is a tool that checks if your Python project's dependencies are correctly declared. It looks for three types of issues:

1. **DEP001**: "Missing dependency" - You're importing something that isn't in your `pyproject.toml` dependencies list
2. **DEP002**: "Unused dependency" - You have a dependency listed but never import it
3. **DEP003**: "Transitive dependency" - You're importing something that should be a direct dependency but is only available transitively (through another package)

## The Problem: Local Modules vs. Package Dependencies

### What are "Local Modules"?

**Local modules** are Python files/modules that are part of your own codebase, not external packages you install.

**Example:**
- `pickleball_pose_analyzer` - This is YOUR code in `libraries/src/pickleball_pose_analyzer/`
- `sam_3d_body` - This is YOUR code in `models/sam-3d-body/sam_3d_body/`
- `tools` - This is YOUR code in `models/sam-3d-body/tools/`

### What are "Package Dependencies"?

**Package dependencies** are external libraries you install from PyPI (or other sources) using `pip` or `uv`.

**Example:**
- `numpy` - External package, you install it: `uv pip install numpy`
- `torch` - External package, you install it: `uv pip install torch`
- `opencv-python` - External package, you install it: `uv pip install opencv-python`

## Why Ignore Local Modules?

### 1. **They're Not Dependencies - They're Your Code!**

When you write:
```python
from pickleball_pose_analyzer import load_sam3d_model
```

You're importing **your own code**, not an external package. You don't need to install it because it's already part of your project!

**Analogy:** It's like having a function in the same file - you don't need to "install" it, it's just there.

### 2. **They Don't Need to be in `pyproject.toml`**

Your `pyproject.toml` dependencies list is for **external packages** you need to install. Local modules are already in your codebase, so they don't belong there.

**Example:**
```toml
# ✅ Correct - external packages
dependencies = [
    "numpy>=1.24.0",
    "torch>=2.0.0",
]

# ❌ Wrong - local modules don't go here
dependencies = [
    "pickleball_pose_analyzer",  # This is YOUR code, not a package!
]
```

### 3. **Deptry Can't Tell the Difference**

Deptry sees `import pickleball_pose_analyzer` and thinks: "This must be an external package that needs to be installed!" But it's actually just your local code.

**This is a false positive** - deptry is flagging something that's not actually a problem.

## Real-World Analogy

Think of it like this:

- **External dependencies** = Tools you buy from a store (hammer, screwdriver) - you need to list them in your shopping list
- **Local modules** = Tools you already have in your garage - you don't need to buy them, they're already yours!

## The Solution: Exclude or Ignore

Since deptry can't automatically distinguish local modules from external packages, we have two options:

### Option 1: Exclude Directories (Recommended)
Exclude entire directories that contain local modules:
```toml
[tool.deptry]
ignore = [
    "models/sam-3d-body/**",  # All code here is local
    "notebooks/**",  # Notebooks have their own local modules
]
```

### Option 2: Ignore Specific Modules
Use `--per-rule-ignores` to tell deptry to ignore specific modules for specific error codes:
```bash
deptry . --per-rule-ignores "DEP001=pickleball_pose_analyzer|sam_3d_body|tools"
```

## When Should You NOT Ignore?

You should **NOT** ignore if:
- It's an external package that you actually need to install
- It's a package that should be in your dependencies but isn't
- It's a transitive dependency that should be direct

**Example of a REAL problem:**
```python
import some_external_library  # This is a real package from PyPI
```

If `some_external_library` isn't in your `pyproject.toml`, that's a **real problem** that deptry should catch!

## Summary

**Why ignore local modules?**
- They're your code, not external dependencies
- They don't need to be installed
- They don't belong in `pyproject.toml` dependencies
- Deptry's warnings about them are false positives

**Is this the right decision?**
✅ **Yes!** It's standard practice to exclude local modules from dependency checking tools.
