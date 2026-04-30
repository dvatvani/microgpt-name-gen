# microgpt-name-gen

A small MicroGPT-based name generator. This project is mostly based on Andrej Karpathy's [microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The main changes are the addition of a nice CLI interface to the program.

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Quickstart

### 1. Install `uv`

This project uses [uv](https://github.com/astral-sh/uv) as package manager.
Installation instructions are [here](https://docs.astral.sh/uv/getting-started/installation/).
It's recommended to install the latest version from [github releases](https://github.com/astral-sh/uv/releases).


Linux/MacOS (bash):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (powershell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you have already installed `uv`, please ensure you're using the latest version by running `uv self update`.

### 2. Clone repository

```bash
git clone https://github.com/dvatvani/microgpt-name-gen
cd microgpt-name-gen
```


### 4. Run initial setup
   ```bash
   just install
   ```

Congratulations! You're now ready to use your new project :sparkles:

## Justfile commands

Run tasks with `just <recipe>`. All commands are defined in-place in `justfile`.

- `install` — Run `sync` and install pre-commit hooks.
- `check` — Check code format and lint with [ruff](https://github.com/astral-sh/ruff) and [ty](https://github.com/astral-sh/ty) for type checking.
- `sync` — Synchronize dependencies with `uv sync`.
<<<<<<< before updating
- `tests` — Run pytest and remove `__pycache__`/`.pyc` files.
- `type-check` — Run type checking with [ty](https://github.com/astral-sh/ty).
=======
- `test` — Run pytest and remove `__pycache__`/`.pyc` files.


>>>>>>> after updating
