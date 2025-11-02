# Migration to uv and pyproject.toml

This document explains the migration from `requirements.txt` to `pyproject.toml` with `uv` for dependency management.

## What Changed?

### Before (requirements.txt)
```bash
pip install -r requirements.txt
```

### After (pyproject.toml + uv)
```bash
uv pip install -e ".[dev]"
```

## Why uv?

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust by Astral (creators of Ruff). Benefits include:

- ⚡ **10-100x faster** than pip
- 🔒 **Deterministic** dependency resolution
- 📦 **Modern** `pyproject.toml` support (PEP 621)
- 🎯 **Compatible** with pip and existing workflows
- 💾 **Disk efficient** with shared package cache

## New Dependency Management

### Project Structure

```toml
# pyproject.toml
[project]
dependencies = [
    # Core dependencies
    "torch>=2.0.0",
    "ultralytics>=8.0.0",
    # ...
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.7.0",
    # ...
]

experiments = [
    "wandb>=0.15.0",
]

gpu = [
    "faiss-gpu>=1.7.4",
]
```

### Installation Options

```bash
# Install core dependencies only
uv pip install -e .

# Install with development tools
uv pip install -e ".[dev]"

# Install with experiments tracking
uv pip install -e ".[experiments]"

# Install with GPU support
uv pip install -e ".[gpu]"

# Install everything
uv pip install -e ".[all]"
```

## Quick Reference

### Common Commands

| Task | Old (pip) | New (uv) |
|------|-----------|----------|
| Install deps | `pip install -r requirements.txt` | `uv pip install -e .` |
| Install dev | `pip install -r requirements-dev.txt` | `uv pip install -e ".[dev]"` |
| Add package | Edit requirements.txt + `pip install` | Edit pyproject.toml + `uv pip install` |
| Update all | `pip install -U -r requirements.txt` | `uv pip install -U -e .` |
| Freeze | `pip freeze > requirements.txt` | `uv pip compile pyproject.toml` |

### Make Commands

```bash
# Install dependencies
make install           # Core dependencies
make install-dev       # With dev tools
make install-all       # Everything

# Generate requirements.txt (for compatibility)
make lock

# Sync dependencies
make sync
```

## Installing uv

### Automatic (via setup.sh)
```bash
./setup.sh
```

### Manual Installation

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**With pip:**
```bash
pip install uv
```

**With pipx:**
```bash
pipx install uv
```

## Virtual Environment Management

### Creating Virtual Environments

```bash
# With uv (faster)
uv venv

# Activate
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# With specific Python version
uv venv --python 3.10
```

### Installing in Virtual Environment

```bash
# Activate environment first
source .venv/bin/activate

# Install project
uv pip install -e ".[dev]"
```

## Adding New Dependencies

### 1. Edit pyproject.toml

```toml
[project]
dependencies = [
    # Add new dependency
    "new-package>=1.0.0",
]
```

### 2. Install

```bash
uv pip install -e .
```

### 3. Generate requirements.txt (optional, for compatibility)

```bash
make lock
# or
uv pip compile pyproject.toml -o requirements.txt
```

## Backward Compatibility

### Generating requirements.txt

For systems that still need `requirements.txt`:

```bash
# Generate from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt

# With Make
make lock
```

### Using pip (fallback)

If you can't use uv, you can still use pip:

```bash
# Generate requirements.txt first
uv pip compile pyproject.toml -o requirements.txt

# Install with pip
pip install -r requirements.txt
```

## Docker Integration

The Dockerfile now uses uv:

```dockerfile
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy pyproject.toml
COPY pyproject.toml .

# Install with uv
RUN uv pip install --system -e .
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv pip install -e ".[dev]"

- name: Run tests
  run: pytest
```

## Troubleshooting

### uv command not found

**Solution:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or source your shell config
source ~/.bashrc  # or ~/.zshrc
```

### Virtual environment issues

**Solution:**
```bash
# Remove old venv
rm -rf .venv

# Create new with uv
uv venv

# Activate and install
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Dependency conflicts

**Solution:**
```bash
# uv provides better error messages
uv pip install -e ".[dev]"

# If conflicts persist, check pyproject.toml version constraints
```

## Performance Comparison

Approximate installation times for SmartPantry dependencies:

| Tool | Time | Notes |
|------|------|-------|
| pip | ~120s | With cache |
| uv | ~8s | 15x faster |

## Migration Checklist

- [x] Create `pyproject.toml`
- [x] Update `setup.sh` to use uv
- [x] Update `Makefile` commands
- [x] Update `Dockerfile`
- [x] Update `.gitignore`
- [x] Update documentation
- [x] Add `.python-version` file
- [x] Keep `requirements.txt` generation option

## Best Practices

1. **Version Pinning**: Use `>=` for flexibility, `==` for reproducibility
2. **Optional Dependencies**: Group by feature (dev, experiments, gpu)
3. **Lock File**: Generate `requirements.txt` for deployment
4. **CI/CD**: Use uv in CI for faster builds
5. **Documentation**: Keep pyproject.toml as source of truth

## Further Reading

- [uv Documentation](https://github.com/astral-sh/uv)
- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [PEP 518 - Specifying Minimum Build System Requirements](https://peps.python.org/pep-0518/)
- [Python Packaging User Guide](https://packaging.python.org/)

## Support

For issues with:
- **uv**: https://github.com/astral-sh/uv/issues
- **SmartPantry**: Open an issue in project repository

