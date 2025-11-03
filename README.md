# 🚀 Skyulf-MLFlow-Library

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-133%20total%20(132%20passing%2C%201%20skipped)-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-56%25-yellow.svg)]()

> **My personal ML toolkit built from years of frustration with scattered libraries and repetitive code**

Hey there! This library started because I got tired of copy-pasting the same preprocessing code across every ML project. You know the drill - load data, handle missing values, encode categories, scale features, train model, save it somewhere... rinse and repeat.

So I built something better. **Skyulf-MLFlow-Library** pulls together the best parts of scikit-learn, pandas, and imbalanced-learn into one clean API. Think of it as the ML utility belt I wish I'd had when I started.

## 💡 Why I Built This

Look, I've been there. You're building your 10th customer churn model and you're writing the same damn preprocessing pipeline again. Or you're trying to remember if you used `OneHotEncoder` or `get_dummies` in that project from six months ago. Or worse - your model works great in notebooks but completely fails in production because the preprocessing doesn't match.

This library solves those headaches:
- **No more repetitive boilerplate** - Write it once, use it everywhere
- **Consistent patterns** - Same API whether you're imputing, encoding, or engineering features  
- **Production-ready from day one** - Bundle transformers with models to avoid train/serve skew
- **Smart defaults** - Things just work without tweaking 20 parameters

I work on this nights and weekends when I hit something annoying in my ML work and think "there's gotta be a better way." If it helps you too, awesome! Pull requests always welcome.

### 🎯 What's Actually Useful Here?

**Real automation that saves time:**
- Domain detection that actually understands your data (e-commerce vs healthcare vs finance)
- Automatically suggests what features to engineer based on data type
- Quality reports that tell you what's broken before your model does

**One API to rule them all:**
- Everything follows the same `.fit()`, `.transform()` pattern (because muscle memory is real)
- Pass DataFrames in, get DataFrames out (no more "wait, is this a numpy array or pandas?")
- Chain operations without fighting type compatibility

**Features I actually use in production:**
- Create ratios, aggregations, and datetime features with simple config instead of custom functions
- Handle high-cardinality categories without blowing up memory (target encoding, hashing)
- Pick the best features automatically from 9 different selection methods
- Save entire pipelines (not just models) so preprocessing works the same in production

**Not vaporware:**
- 133 tests covering the important stuff
- Type hints everywhere so your IDE actually helps
- Been running in real projects for months
- Decent error messages when things go wrong

---

## ✨ Features

### 🎯 **Smart Data Understanding** (Unique!)
- **Domain Analyzer**: Automatically detects if your data is e-commerce, healthcare, finance, time-series, etc.
- **Text Insights**: Analyzes text columns and provides recommendations
- **Quality Reports**: Generates comprehensive data quality assessments
- **EDA Automation**: Smart exploratory data analysis based on domain

### 🔧 **Advanced Feature Engineering**
- **FeatureMath**: Create complex features with operations like:
  ```python
  {'type': 'arithmetic', 'method': 'add', 'columns': ['price', 'tax'], 'output': 'total'}
  {'type': 'ratio', 'numerator': ['sales'], 'denominator': ['cost'], 'output': 'profit_margin'}
  {'type': 'datetime', 'method': 'extract_hour', 'columns': ['timestamp']}
  ```
- **SmartBinning**: Multiple strategies (equal_width, equal_frequency, kmeans, custom)
- **Feature Selection**: 9+ methods (SelectKBest, RFE, variance threshold, model-based)
- **Polynomial Features**: Automatic interaction and polynomial term generation

### 🏷️ **Comprehensive Encoding**
- One-Hot Encoding (with sparse support)
- Label Encoding (with unseen category handling)
- Ordinal Encoding (preserves order)
- Target Encoding (for high-cardinality features)
- Hash Encoding (for memory efficiency)

### 🧹 **Preprocessing Made Easy**
- **Imputation**: Mean, median, mode, constant, forward-fill, backward-fill
- **Scaling**: Standard, MinMax, Robust, MaxAbs
- **Cleaning**: Drop missing (rows/columns), remove duplicates, outlier detection
- **Sampling**: SMOTE, over-sampling, under-sampling (requires imbalanced-learn)

### 📊 **Model Management** (Production-Ready!)
- **Model Registry**: SQLite-based versioning system
- **Metrics Calculator**: Comprehensive evaluation for classification & regression
- **Metadata Tracking**: Store hyperparameters, tags, descriptions
- **Easy Deployment**: Load models by name or version

### 💾 **Data Ingestion**
- Multi-format support: CSV, Excel, JSON, Parquet, SQL
- Automatic format detection
- Consistent save/load interface
- Built-in error handling

---

## 📦 Installation

### From Source (Current)
```bash
# Clone the repository
git clone https://github.com/flyingriverhorse/skyulf_mlflow_library.git
cd skyulf_mlflow_library

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Requirements
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

## 📚 Comprehensive Examples

Check out the [`examples/`](examples/) folder for detailed, runnable examples:

- **[01_basic_usage.py](examples/01_basic_usage.py)** - Getting started with the basics
- **[02_comprehensive_pipeline.py](examples/02_comprehensive_pipeline.py)** - Full ML workflow
- **[04_feature_math.py](examples/04_feature_math.py)** - Advanced feature engineering ⭐
- **[05_smart_binning.py](examples/05_smart_binning.py)** - Intelligent binning strategies
- **[09_feature_selection.py](examples/09_feature_selection.py)** - Feature selection methods
- **[10_full_library_showcase.py](examples/10_full_library_showcase.py)** - Everything in one place ⭐
- **[11_domain_analyzer.py](examples/11_domain_analyzer.py)** - Smart domain detection ⭐
- **[12_eda_overview.py](examples/12_eda_overview.py)** - Automated EDA capabilities

---

## 🏗️ Architecture

```
skyulf-mlflow/
├── skyulf_mlflow/
│   ├── core/               # Base classes, types, exceptions
│   ├── data_ingestion/     # Data loading utilities
│   ├── preprocessing/      # Imputation, scaling, cleaning, sampling
│   ├── features/          # Encoding and transformations
│   │   ├── encoding/      # OneHot, Label, Ordinal, Target, Hash
│   │   └── transform/     # FeatureMath, Polynomial, Binning
│   ├── modeling/          # Metrics, model registry
│   ├── pipeline/          # Pipeline system
│   └── utils/             # Splitting and utilities
├── examples/              # Example scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
```

---

## 🧪 Testing & Quality

- **133 Tests**: Comprehensive test coverage
- **56% Coverage**: Core functionality tested  
- **All Tests Passing**: ✅ 132/133 (1 skipped due to optional dependency)
- **Type Hints**: Throughout the codebase
- **Error Handling**: Proper exceptions and validation

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=skyulf_mlflow --cov-report=term
```

---

## 📖 Documentation

### 📂 Structure
```
docs/
├── api/                          # API documentation
│   ├── feature_engineering.md
│   ├── domain_analysis.md
│   └── eda.md
├── CONTRIBUTING.md               # How to contribute
└── PYPI_PUBLISHING.md           # Publishing guide

examples/
├── 01_basic_usage.py
├── 04_feature_math.py           # ⭐ Advanced features
├── 10_full_library_showcase.py # ⭐ Everything
└── 11_domain_analyzer.py        # ⭐ Unique feature
```

### 🔗 Quick Links
- [Contributing Guide](skyulf_mlflow/CONTRIBUTING.md)
- [API Documentation](docs/api/)
- [Examples](examples/)

---

## 🗺️ Roadmap

This is an **ongoing side project**. Future improvements I'm planning:

**Want to help?** Contributions are very welcome! See [CONTRIBUTING.md](skyulf_mlflow/CONTRIBUTING.md)

---

## 💭 Philosophy

This library is built on these principles:

1. **🎯 Practicality Over Perfection**: Features that solve real problems
2. **🤝 Ease of Use**: Consistent API, good defaults, clear errors
3. **🔧 Composability**: Mix and match components freely
4. **📚 Learning by Doing**: Each feature comes from actual ML work
5. **🌱 Continuous Improvement**: Always evolving based on experience

---

## 🤝 Contributing

I built this for myself and my projects, but I'd love your input!

**Ways to contribute:**
- 🐛 Report bugs or issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add more tests
- ⭐ Star the repo if you find it useful!

See [CONTRIBUTING.md](skyulf_mlflow/CONTRIBUTING.md) for:
- How to add new features
- Testing guidelines
- Code style conventions
- Pull request process

### Development Setup
```bash
# Clone repository
git clone https://github.com/flyingriverhorse/skyulf_mlflow_library.git
cd skyulf_mlflow_library

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check coverage
pytest tests/ --cov=skyulf_mlflow --cov-report=html
```

---

## ⚠️ Current Status

**Version**: 0.1.1 (Alpha)

This is a **work in progress**. I update it whenever I:
- Build a new ML project and need a feature
- Find a better way to do something
- Get inspired by other libraries
- Have some free time on weekends 😄

**What this means:**
- ✅ Core functionality is solid and tested
- ✅ API is relatively stable
- ⚠️ Some features are still evolving
- ⚠️ Documentation is improving continuously
- 🚧 New features added regularly

**Use it if:**
- You want practical ML tools
- You like the "batteries included" approach
- You don't mind some rough edges
- You want to contribute and help shape it

---

## 📊 Stats

- **Tests**: 133 (132 passing)
- **Coverage**: 56%
- **Examples**: 12
- **Python**: 3.8 - 3.12
- **Dependencies**: Minimal (mostly stdlib + scientific stack)

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

### Latest: v0.1.1 (2025-11-01)
- ✅ Added EDA module with domain analysis
- ✅ Added feature selection with multiple strategies
- ✅ Added comprehensive test suite (133 tests, 56% coverage)
- ✅ Added 12 runnable examples
- ✅ Added CONTRIBUTING guide
- ✅ Initial alpha release
- ✅ Core preprocessing and feature engineering
- ✅ Model registry and metrics
- ✅ Basic documentation

---

## 🙏 Acknowledgments

This library stands on the shoulders of giants:

- **[scikit-learn](https://scikit-learn.org/)**: The foundation of ML in Python
- **[pandas](https://pandas.pydata.org/)**: Data manipulation made easy
- **[numpy](https://numpy.org/)**: Numerical computing powerhouse
- **[imbalanced-learn](https://imbalanced-learn.org/)**: Handling class imbalance
- **[scipy](https://scipy.org/)**: Scientific computing tools

And countless blog posts, Stack Overflow answers, and GitHub repos that taught me better ways to do things. Thank you all! 🙏

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

**TL;DR**: Use it however you want, just don't sue me if something breaks 😊

Copyright (c) 2025 Murat Unsal

---

## 📧 Contact & Support

**Author**: Murat Unsal

- 📫 Email: murath.unsal@unsal.com
- 🐙 GitHub: [flyingriverhorse/MLops](https://github.com/flyingriverhorse/skyulf_mlflow_library)
- 💬 Issues: [Report a bug or request a feature](https://github.com/flyingriverhorse/skyulf_mlflow_library/issues)

---

## ⭐ Star History

If this library helps you, consider giving it a star! It motivates me to keep improving it 🌟

---

## 🎉 Final Words

This is a **passion project** born from real ML work. I built the tools I wished existed when I started doing data science. If it helps you too, that's awesome!

**Remember:**
- Start simple, add complexity when needed
- Good features > Many features
- Documentation is love ❤️
- Tests are your safety net 🛡️

**Happy Machine Learning! 🚀**

---

<div align="center">

Made with ☕ and 💻 during late nights and weekends

**If you find this useful, give it a ⭐ and spread the word!**

</div>
