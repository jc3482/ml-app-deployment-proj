# uv Quick Reference for SmartPantry

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"
```

## Setup Project

```bash
# Quick setup
./setup.sh

# Or manually
uv venv                          # Create virtual environment
source .venv/bin/activate        # Activate
uv pip install -e ".[dev]"       # Install with dev dependencies
```

## Common Commands

```bash
# Install variations
uv pip install -e .              # Core dependencies only
uv pip install -e ".[dev]"       # With development tools
uv pip install -e ".[experiments]" # With experiment tracking
uv pip install -e ".[gpu]"       # With GPU support
uv pip install -e ".[all]"       # Everything

# Package management
uv pip install package-name      # Install package
uv pip install -U package-name   # Upgrade package
uv pip uninstall package-name    # Remove package
uv pip list                      # List installed packages

# Environment
uv venv                          # Create venv
uv venv --python 3.10            # With specific Python
uv pip sync                      # Sync to exact versions

# Generate requirements.txt
uv pip compile pyproject.toml -o requirements.txt
```

## Make Commands

```bash
make install          # Install core dependencies
make install-dev      # Install with dev tools
make install-all      # Install all optional dependencies
make lock             # Generate requirements.txt
make sync             # Sync dependencies
```

## Why uv is Fast

- Written in Rust (10-100x faster than pip)
- Parallel downloads
- Efficient dependency resolution
- Shared package cache
- No subprocess overhead

## Tips

1. Always activate venv first: `source .venv/bin/activate`
2. Use `make install-dev` for development
3. Run `make lock` before committing to update requirements.txt
4. Use `uv pip list` to check installed packages
5. `uv venv` is faster than `python -m venv`

## Troubleshooting

```bash
# Command not found
export PATH="$HOME/.cargo/bin:$PATH"

# Reinstall everything
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Check uv version
uv --version
```

## More Info

- uv docs: https://github.com/astral-sh/uv
- Full migration guide: `docs/UV_MIGRATION.md`
