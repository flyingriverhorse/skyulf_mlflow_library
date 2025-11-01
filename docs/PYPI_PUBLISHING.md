# Publishing Skyulf MLflow to PyPI

This guide explains how to build and publish Skyulf MLflow to the Python Package Index (PyPI).

---

## 📋 Prerequisites

1. **PyPI Account**
   - Create account at [https://pypi.org/account/register/](https://pypi.org/account/register/)
   - Verify your email address

2. **TestPyPI Account** (for testing)
   - Create account at [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)

3. **Install Build Tools**
   ```bash
   pip install --upgrade build twine
   ```

4. **API Tokens** (Recommended over passwords)
   - Go to PyPI account settings
   - Generate an API token for uploading
   - Save it securely

---

## 🔨 Step 1: Prepare Your Package

### 1.1 Update Version Number

Edit `pyproject.toml`:
```toml
[project]
name = "skyulf-mlflow"
version = "0.1.1"  # Update this!
```

### 1.2 Update CHANGELOG.md

Add release notes for the new version:
```markdown
## [0.1.1] - 2025-11-01

### Added
- DomainAnalyzer and infer_domain helper promoted to the public API
- New API documentation and examples for dataset domain inference

### Changed
- Updated publishing assets and metadata for the 0.1.1 release
```

### 1.3 Verify pyproject.toml

Ensure all metadata is correct:
```toml
[project]
name = "skyulf-mlflow"
version = "0.1.1"
description = "A comprehensive machine learning library..."
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Murat", email = "murath.unsal@unsal.com"}
]
keywords = [
    "machine-learning",
    "data-science",
    "feature-engineering",
    ...
]
dependencies = [
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
    "rapidfuzz>=3.0.0",
]
```

### 1.4 Check Package Structure

```
skyulf-mlflow/
├── skyulf_mlflow_library/        # Main package
│   ├── __init__.py
│   ├── core/
│   ├── preprocessing/
│   ├── features/
│   ├── modeling/
│   ├── pipeline/
│   ├── utils/
│   ├── eda/               # Lightweight EDA utilities and domain analyzer
│   └── data_ingestion/
├── tests/                # Tests
├── examples/             # Example scripts
├── pyproject.toml        # Build configuration
├── README.md             # Package description
├── LICENSE               # License file
├── CHANGELOG.md          # Version history
└── MANIFEST.in           # Include non-Python files
```

---

## 🧪 Step 2: Test Locally

### 2.1 Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/ dist/ *.egg-info/

# On Windows PowerShell:
Remove-Item -Recurse -Force build, dist, *.egg-info
```

### 2.2 Build the Package

```bash
python -m build
```

This creates two files in `dist/`:
- `skyulf_mlflow_library-0.1.1-py3-none-any.whl` (wheel format)
- `skyulf-mlflow-0.1.1.tar.gz` (source distribution)

### 2.3 Test Installation Locally

```bash
# Create a new virtual environment for testing
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from local wheel
pip install dist/skyulf_mlflow_library-0.1.1-py3-none-any.whl

# Test import
python -c "import skyulf_mlflow_library; print(skyulf_mlflow_library.__version__)"

# Run a quick test
python -c "from skyulf_mlflow_library.preprocessing import StandardScaler; print('Success!')"

# Execute the full showcase example
python examples/10_full_library_showcase.py

# Deactivate and remove test environment
deactivate
rm -rf test_env
```

---

## 🧪 Step 3: Upload to TestPyPI (Optional but Recommended)

### 3.1 Create .pypirc Configuration

Create or edit `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-production-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

**Security Note**: Keep this file private! Set permissions:
```bash
chmod 600 ~/.pypirc
```

### 3.2 Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

### 3.3 Test Installation from TestPyPI

```bash
# Create fresh environment
python -m venv testpypi_env
source testpypi_env/bin/activate

# Install from TestPyPI (may need to specify index for dependencies)
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    skyulf-mlflow

# Test
python -c "import skyulf_mlflow_library; print('TestPyPI installation successful!')"

# Cleanup
deactivate
rm -rf testpypi_env
```

---

## 🚀 Step 4: Upload to PyPI (Production)

### 4.1 Final Checks

- ✅ All tests passing
- ✅ Documentation updated
- ✅ CHANGELOG.md updated
- ✅ Version bumped in pyproject.toml
- ✅ Tested on TestPyPI successfully
- ✅ README.md looks good
- ✅ Git committed and tagged

### 4.2 Create Git Tag

```bash
git add .
git commit -m "Release v0.1.1"
git tag -a v0.1.1 -m "Version 0.1.1 - Domain analyzer updates"
git push origin main
git push origin v0.1.1
```

### 4.3 Upload to PyPI

```bash
python -m twine upload dist/*
```

Or with explicit token:
```bash
python -m twine upload -u __token__ -p pypi-your-api-token dist/*
```

### 4.4 Verify Upload

1. Visit https://pypi.org/project/skyulf-mlflow/
2. Check that version appears correctly
3. Verify README renders properly
4. Check that files are present

---

## 🎉 Step 5: Post-Publication

### 5.1 Test Installation from PyPI

```bash
# Fresh environment
python -m venv pypi_test_env
source pypi_test_env/bin/activate

# Install from PyPI
pip install skyulf-mlflow

# Test
python -c "import skyulf_mlflow_library; print(f'Version {skyulf_mlflow_library.__version__} installed!')"

# Cleanup
deactivate
rm -rf pypi_test_env
```

### 5.2 Update Documentation

- Update README badges with PyPI version
- Update documentation links
- Announce on social media/forums

### 5.3 Create GitHub Release

1. Go to GitHub repository
2. Click "Releases" → "Create a new release"
3. Select tag v0.1.1
4. Title: "v0.1.1 - Domain Analyzer Release"
5. Description: Copy from CHANGELOG.md
6. Attach distribution files (optional)
7. Publish release

---

## 🔄 Updating an Existing Package

For subsequent releases:

1. **Make changes** to code
2. **Update version** in pyproject.toml (follow semantic versioning)
3. **Update CHANGELOG.md**
4. **Commit changes**: `git commit -am "Prepare v0.5.1"`
5. **Tag version**: `git tag v0.5.1`
6. **Clean build**: `rm -rf dist/`
7. **Build**: `python -m build`
8. **Upload**: `python -m twine upload dist/*`
9. **Push**: `git push && git push --tags`

---

## 📊 Semantic Versioning Guide

Use [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH

- **MAJOR**: Incompatible API changes (e.g., 1.0.0 → 2.0.0)
- **MINOR**: New functionality (backwards-compatible) (e.g., 0.1.0 → 0.2.0)
- **PATCH**: Bug fixes (backwards-compatible) (e.g., 0.1.0 → 0.1.1)

Examples:
- `0.1.0` → `0.1.1` : Bug fixes
- `0.1.0` → `0.2.0` : New features added
- `0.1.0` → `1.0.0` : Stable release, or breaking changes

---

## 🔐 Security Best Practices

1. **Never commit** `.pypirc` or API tokens to Git
2. **Use API tokens** instead of passwords
3. **Scope tokens** to specific projects when possible
4. **Rotate tokens** periodically
5. **Use TestPyPI** for testing before production
6. **Add `.pypirc` to `.gitignore`**

---

## ❗ Troubleshooting

### Error: "File already exists"
- You cannot re-upload the same version
- Bump version number and rebuild
- Or delete from PyPI (if just uploaded)

### Error: "Invalid distribution"
- Check pyproject.toml syntax
- Ensure README.md exists
- Verify package structure

### Error: "Authentication failed"
- Check API token is correct
- Ensure token has upload permissions
- Verify .pypirc format

### README not rendering
- Check markdown syntax
- Ensure relative links work
- Preview on GitHub first

---

## 📝 Checklist for Publishing

Before publishing, verify:

- [ ] All tests pass (`pytest tests/`)
- [ ] Version bumped in pyproject.toml
- [ ] CHANGELOG.md updated
- [ ] README.md updated
- [ ] Documentation current
- [ ] Examples work
- [ ] Dependencies correct
- [ ] License file present
- [ ] Git committed
- [ ] Git tagged
- [ ] Built successfully (`python -m build`)
- [ ] Tested locally
- [ ] (Optional) Tested on TestPyPI
- [ ] Uploaded to PyPI
- [ ] Verified on PyPI website
- [ ] Installation tested
- [ ] GitHub release created

---

## 📚 Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

---

