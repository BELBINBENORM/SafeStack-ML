import os
import gc
import numpy as np
import joblib
import cloudpickle
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import contextlib
try:
    from sklearn.utils import parallel_config
except ImportError:
    @contextlib.contextmanager
    def parallel_config(**kwargs):
        yield
from joblib import Parallel, delayed
class SafeStackingClassifier(StackingClassifier):
    """
    A robust Stacking Classifier designed for memory efficiency and process safety.
    Features automated artifact saving, memory cleanup, and version-mismatch patching.
    """
    
    def __init__(self, estimators, final_estimator=None, base_path="./", cv=5, n_jobs=-1, verbose=1, **kwargs):
        super().__init__(estimators=estimators, final_estimator=final_estimator, cv=cv, n_jobs=n_jobs, verbose=verbose, **kwargs)
        self.base_path = base_path
        self.parent_pid = os.getpid()

    def _log(self, message, required_level=1):
        if self.verbose >= required_level:
            if required_level == 1 and os.getpid() != self.parent_pid:
                return
            print(message, flush=True)

    def fit(self, X, y, sample_weight=None):
        self.parent_pid = os.getpid()
        self.skf_ = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        n_samples, n_estimators = X.shape[0], len(self.estimators)
        
        self._log(f"🚀 [INIT] Engine: {n_samples} rows | {n_estimators} Experts | CV: {self.cv}", 1)
        self._log(f"📂 [PATH] Searching Base: {self.base_path} | Local: ./", 1)

        # THE DISTRIBUTED WAVE LOOP
        with parallel_config(backend='loky', n_jobs=self.n_jobs, temp_folder='/dev/shm'):
            for fold_idx, (train_idx, val_idx) in enumerate(self.skf_.split(X, y)):
                self._log(f"\n🌊 [WAVE {fold_idx + 1}/{self.cv}] Checking artifacts...", 1)
                
                if hasattr(X, 'iloc'):
                    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    X_tr, X_va = X[train_idx], X[val_idx]
                    y_tr, y_va = y[train_idx], y[val_idx]

                _ = Parallel()(
                    delayed(self._fit_save_base_and_oof)(name, est, X_train, y_train, X_val, y_val, fold_idx)
                    for name, est in self.estimators
                )
                
                del X_train, X_val, y_train, y_val
                gc.collect()

        # REASSEMBLY
        self._log("\n🏗️ [REASSEMBLE] Constructing OOF Matrix from Disk...", 1)
        meta_features = np.zeros((n_samples, n_estimators), dtype='float32')
      
        for fold_idx, (train_idx, val_idx) in enumerate(self.skf_.split(X, y)):
            for i, (name, _) in enumerate(self.estimators):
                oof_file = f"oof_{name}_fold_{fold_idx}.pkl"
                final_path = oof_file if os.path.exists(oof_file) else os.path.join(self.base_path, oof_file)
                meta_features[val_idx, i] = joblib.load(final_path)

        # FIT THE JUDGE
        self._log(f"⚖️ [META] Fitting Judge on OOF Matrix: {meta_features.shape}", 1)
        self.final_estimator_ = clone(self.final_estimator).fit(meta_features, y)
        
        # FINAL FULL TRAINING
        self._log("📥 [FINAL] Training/Loading Experts on 100% data...", 1)
        with parallel_config(backend='loky', n_jobs=self.n_jobs, temp_folder='/dev/shm'):
            self.estimators_ = Parallel()(
                delayed(self._fit_final_full_model)(name, est, X, y)
                for name, est in self.estimators
            )

        # THE METADATA CHECKLIST
        self.classes_ = np.unique(y)
        self.stack_method_ = ['predict_proba'] * len(self.estimators_)
        self.named_estimators_ = {name: est for name, est in zip([e[0] for e in self.estimators], self.estimators_)}
        
        # Patch LabelEncoder for version compatibility
        le = LabelEncoder()
        le.classes_ = self.classes_
        self._label_encoder = le

        # Patch Judge for version mismatch
        if not hasattr(self.final_estimator_, 'multi_class'):
            self.final_estimator_.multi_class = 'auto'
        if not hasattr(self.final_estimator_, 'classes_'):
            self.final_estimator_.classes_ = self.classes_

        self._log("\n✨ [COMPLETE] Stack fully assembled and injected with metadata.", 1)
        return self

    def _fit_save_base_and_oof(self, name, est, X_tr, y_tr, X_val, y_val, fold_idx):
        model_file = f"model_{name}_fold_{fold_idx}.pkl"
        oof_file = f"oof_{name}_fold_{fold_idx}.pkl"

        if (self.base_path != "./" and os.path.exists(os.path.join(self.base_path, model_file))) or os.path.exists(model_file):
            self._log(f"⏭️ [BYPASS] {name} | Fold {fold_idx + 1} exists.", 2)
            return True

        model = clone(est)
        self._log(f"🔨 [Expert Fit] {name} | Fold {fold_idx + 1}", 2)
        model.fit(X_tr, y_tr)
        
        oof_preds = model.predict_proba(X_val)[:, 1]
        joblib.dump(oof_preds, oof_file)
        
        with open(model_file, "wb") as f:
            cloudpickle.dump(model, f)
        self._log(f"✅ [Expert Saved] {name} | Fold {fold_idx + 1}", 2)
        
        del model, oof_preds
        gc.collect()
        self._log(f"🧹 [CLEAN] {name} | Fold {fold_idx + 1} memory released.", 2)
        return True

    def _fit_final_full_model(self, name, est, X, y):
        final_file = f"final_full_model_{name}.pkl"
        search_path = final_file if os.path.exists(final_file) else os.path.join(self.base_path, final_file)

        if os.path.exists(search_path):
            self._log(f"⏭️ [BYPASS] {name} Full Model exists.", 2)
            with open(search_path, "rb") as f:
                return cloudpickle.load(f)
                
        self._log(f"🚀 [Full Fit] {name} starting...", 2)
        full_model = clone(est).fit(X, y)
        with open(final_file, "wb") as f:
            cloudpickle.dump(full_model, f)
            
        self._log(f"✅ [Full Fit] {name} completed.", 2)
        return full_model
