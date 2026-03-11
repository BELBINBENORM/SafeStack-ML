# SafeStack-ML

**SafeStack-ML** is a production-grade implementation of a Stacking Classifier engineered for high-stability machine learning workflows. It addresses critical bottlenecks in ensemble learning—such as memory exhaustion, process crashes during long-running fits, and environment version mismatches—by implementing a disk-persistent "Wave" architecture.

## 🚀 Key Engineering Features

* **Artifact Persistence (Check-pointing):** Automatically saves Out-of-Fold (OOF) predictions and model experts to disk using `joblib` and `cloudpickle`. If a training process is interrupted, the engine detects existing files and bypasses completed work to resume training.
* **Memory Management:** Implements aggressive memory cleanup using `gc.collect()` and explicit object deletion (`del`) after each training fold to prevent memory leaks during large-scale ensemble fits.
* **Optimized Parallelism:** Utilizes the `loky` backend with shared memory optimization via the `/dev/shm` temporary folder to accelerate parallel execution while minimizing I/O overhead.
* **Compatibility Patching:** Includes built-in metadata injection and patches for `LabelEncoder` and final estimators to ensure model serialization and prediction work across varying library versions.

## 🛠️ Installation

### For Local Development:
```bash
# Clone the repository
git clone [https://github.com/BELBINBENORM/SafeStack-ML.git](https://github.com/BELBINBENORM/SafeStack-ML.git)
cd SafeStack-ML

# Install via setup.py
pip install 
```

### For Kaggle / Colab / Jupyter:
Run this in a cell to install the library directly from GitHub:

```python
!pip install git+[https://github.com/BELBINBENORM/SafeStack-ML.git](https://github.com/BELBINBENORM/SafeStack-ML.git)
```
## 💻 Quick Start

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from safe_stack import SafeStackingClassifier

# 1. Generate Sample Data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define your experts
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

# 3. Initialize the Safe Stack
# Artifacts will be saved to './model_artifacts/' to ensure memory safety
clf = SafeStackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1,
    base_path="./model_artifacts/"
)

# 4. Fit with automated checkpointing and memory safety
clf.fit(X_train, y_train)

# 5. Predict
predictions = clf.predict(X_test)
print(f"Model Stacked Predictions: {predictions[:10]}")
```

## 📑 Technical Architecture

* **The Distributed Wave Loop:** Training is executed in "waves" (folds), where each expert is fitted and its OOF predictions are immediately flushed to disk to maintain a low RAM footprint.
* **Reassembly:** The meta-features are reconstructed from disk artifacts, ensuring that the primary process does not hold all model weights in memory simultaneously.
* **Safe Serialization:** Uses `cloudpickle` to handle custom classes and complex objects that standard pickling often fails to capture, ensuring deployment reliability.

## ⚖️ License

Licensed under the **Apache License, Version 2.0**. This project is open for commercial and professional use.
