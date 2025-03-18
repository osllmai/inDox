# Scientific Data

This guide covers scientific data format processors supported in indoxArcg, focusing on numerical datasets and serialized objects.

---

## Supported Loaders

### 1. Scipy MAT Loader

**Best for**: MATLAB data file interoperability

#### Features

- MATLAB v7.3+ format support
- Variable-wise document generation
- Metadata filtering

```python
from indoxArcg.data_loaders import Scipy

loader = Scipy(exclude_vars=["__header__", "__version__"])
docs = loader.load("experiment.mat")
```

---

### 2. Joblib Loader

**Best for**: Python object serialization

#### Features

- Large NumPy array handling
- Compression support
- Safe deserialization

```python
from indoxArcg.data_loaders import Joblib

loader = Joblib(compression="zlib", safe_mode=True)
docs = loader.load("model.pkl")
```

---

## Comparison Table

| Feature          | Scipy MAT        | Joblib         |
| ---------------- | ---------------- | -------------- |
| File Formats     | .mat             | .pkl, .joblib  |
| Data Types       | MATLAB variables | Python objects |
| Compression      | ❌               | ✅ (multiple)  |
| Security         | Basic            | Sandboxed      |
| Max File Size    | 2GB              | 10GB           |
| Parallel Loading | ❌               | ✅             |

---

## Installation

```bash
pip install scipy joblib indoxArcg
```

---

## Basic Usage

### MATLAB Data Handling

```python
from indoxArcg.data_loaders import Scipy

# Load specific variables
docs = Scipy(var_names=["sensor_data", "timestamps"]).load("lab.mat")
```

### Serialized Object Loading

```python
from indoxArcg.data_loaders import Joblib

# Load with memory mapping
docs = Joblib(mmap_mode="r").load("large_array.npy")
```

---

## Advanced Configuration

### Custom Variable Processing

```python
def mat_processor(var_name, var_data):
    return f"{var_name}: {var_data.shape}"

loader = Scipy(var_transform=mat_processor)
```

### Safe Deserialization

```python
from indoxArcg.data_loaders import Joblib

loader = Joblib(
    allowed_classes=["numpy.ndarray", "pandas.DataFrame"],
    max_buffer_size=1e9  # 1GB limit
)
```

---

## Troubleshooting

### Common Issues

1. **MATLAB Version Mismatch**

   ```python
   Scipy(matlab_compat="v7.3").load("legacy.mat")
   ```

2. **Unsafe Pickle Files**

   ```python
   Joblib(safe_mode=True).load("untrusted.pkl")
   ```

3. **Large MAT Files**

   ```python
   Scipy(chunk_size="1GB").load("big_data.mat")
   ```

4. **Compression Errors**
   ```python
   Joblib(compression="lz4", fix_imports=False).load("compressed.joblib")
   ```

---
