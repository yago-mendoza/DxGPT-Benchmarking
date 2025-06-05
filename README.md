# DxGPT Latitude Bench

ML/NLP analysis toolkit for medical diagnostics with ICD-10 support.

## 🚀 Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # Linux/Mac

# 2. Install dependencies
# For CPU-only PyTorch (recommended)
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# Install project in editable mode
py -m pip install -e .
```

> 💡 The virtual environment `.venv` becomes your isolated Python world:
> - `pip install` targets `.venv/Lib/site-packages/`
> - Python looks here first for imports
> - Keep your system Python clean

## 📦 Project Structure

```
utils/              # Core utilities package
├── icd10/         # ICD-10 taxonomy tools
└── services/      # ML/NLP services
    └── bert_similarity.py
```

## 🛠️ Development

```bash
# Install development tools
py -m pip install -e .[dev]

# Run tests
pytest

# Code quality
black .            # Format code
flake8            # Lint code
mypy .            # Type check
```

## 🔧 Under the Hood

When you run `pip install -e .`:
1. pip reads `pyproject.toml` from project root
2. Installs all dependencies to `.venv/Lib/site-packages/`
3. Creates symlinks to your code instead of copying
   ```
   .venv/Lib/site-packages/
   ├── torch, transformers, etc.  # Regular installs
   └── utils -> ../../../utils    # Symlink to your code
   ```

> 💡 The `-e` flag enables "editable mode":
> - Changes to your code take effect immediately
> - No need to reinstall after edits
> - Perfect for development

> 🧹 About `.egg-info/`:
> - Generated during installation
> - Safe to delete (regenerates as needed)
> - Add to `.gitignore`

## 📚 Documentation

For detailed API documentation and examples, see:
- [ICD-10 Taxonomy](utils/icd10/README.md)
- [BERT Similarity](utils/services/README.md)