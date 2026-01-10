# src/cache_utils.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# -----------------------
# Hash / IO helpers
# -----------------------
def config_hash(config: Dict[str, Any], length: int = 10) -> str:
    s = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:length]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_joblib_dump(obj: Any, path: Path, compress: int = 0) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp, compress=compress)
    tmp.replace(path)


def csv_source_signature(csv_path: Path, head_bytes: int = 1_000_000) -> Dict[str, Any]:
    """
    Signature để invalid cache khi file CSV thay đổi.
    Dùng: size + mtime_ns + hash 1MB đầu.
    """
    st = csv_path.stat()
    with csv_path.open("rb") as f:
        head = f.read(head_bytes)
    head_hash = hashlib.md5(head).hexdigest()
    return {
        "name": csv_path.name,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "head_md5": head_hash,
    }


# -----------------------
# Config dataclasses
# -----------------------
@dataclass(frozen=True)
class CleanConfig:
    label_column: str = "label"
    clean_version: int = 1
    dtype: str = "float32"
    nan_policy: str = "fill0"  # "fill0" | "drop_row"


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    min_per_class: int = 5


@dataclass(frozen=True)
class NormConfig:
    method: str = "minmax"  # "minmax" | "zscore"
    feature_range: Tuple[float, float] = (0.0, 1.0)
    clip: bool = True


@dataclass(frozen=True)
class OptimConfig:
    center_init: str = "ver6"
    n_clusters: int = 2
    metric: str = "L2_squared"
    theta_range: Tuple[float, float, float] = (0.01, 0.5, 0.01)
    alpha_range: Tuple[float, float, float] = (0.01, 0.5, 0.01)


# -----------------------
# Cache manager (A-D-B-E)
# -----------------------
class CacheManager:
    def __init__(self, csv_path: str | Path, cache_root: str = "cache"):
        self.csv_path = Path(csv_path)
        self.source_sig = csv_source_signature(self.csv_path)

        self.base_dir = ensure_dir(Path(cache_root) / self.csv_path.stem)
        self.dir_A = ensure_dir(self.base_dir / "level_A_raw")
        self.dir_D = ensure_dir(self.base_dir / "level_D_splits")
        self.dir_B = ensure_dir(self.base_dir / "level_B_norm")
        self.dir_E = ensure_dir(self.base_dir / "level_E_optim")

    # --------
    # Level A
    # --------
    def load_or_build_clean(self, cfg: CleanConfig) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Returns:
            X: float32 array
            y: raw labels (string/int)
            clean_hash: hash key
        """
        cfg_dict = asdict(cfg)
        key = {"cfg": cfg_dict, "source_sig": self.source_sig}
        h = config_hash(key)

        x_path = self.dir_A / f"X_clean_{h}.joblib"
        y_path = self.dir_A / f"y_{h}.joblib"
        meta_path = self.dir_A / f"meta_{h}.json"

        if x_path.exists() and y_path.exists() and meta_path.exists():
            return joblib.load(x_path), joblib.load(y_path), h

        # Build
        df = pd.read_csv(self.csv_path, low_memory=False)

        if cfg.label_column not in df.columns:
            raise ValueError(
                f"Không thấy cột label='{cfg.label_column}'. "
                f"Các cột hiện có (10 cột đầu): {list(df.columns[:10])}"
            )

        y = df[cfg.label_column].to_numpy()
        X_df = df.drop(columns=[cfg.label_column])

        # Giữ numeric; nếu CSV có numeric nhưng bị đọc thành object (do dirty),
        # bạn có thể bật convert ở đây (tốn thời gian hơn):
        X_num = X_df.select_dtypes(include=[np.number])

        X = X_num.to_numpy().astype(np.float32, copy=False)

        if cfg.nan_policy == "fill0":
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        elif cfg.nan_policy == "drop_row":
            good = np.isfinite(X).all(axis=1)
            X = X[good]
            y = y[good]
        else:
            raise ValueError(f"nan_policy không hợp lệ: {cfg.nan_policy}")

        meta = {
            "source_sig": self.source_sig,
            "cfg": cfg_dict,
            "shape": [int(X.shape[0]), int(X.shape[1])],
            "label_column": cfg.label_column,
            "numeric_columns": list(X_num.columns),
        }

        _atomic_joblib_dump(X, x_path, compress=0)   # compress=0 để load nhanh
        _atomic_joblib_dump(y, y_path, compress=0)
        _atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
        return X, y, h

    # --------
    # Level D
    # --------
    def load_or_build_split(
        self, y: np.ndarray, clean_hash: str, cfg: SplitConfig
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Trả về:
            train_idx_global, test_idx_global, split_hash
        """
        cfg_dict = asdict(cfg)
        key = {"cfg": cfg_dict, "clean_hash": clean_hash}
        h = config_hash(key)

        path = self.dir_D / f"split_{h}.npz"
        if path.exists():
            d = np.load(path)
            return d["train_idx"], d["test_idx"], h

        # Filter classes with >= min_per_class
        vals, counts = np.unique(y, return_counts=True)
        keep_vals = set(vals[counts >= cfg.min_per_class])
        keep_idx = np.where(np.isin(y, list(keep_vals)))[0]
        y_filtered = y[keep_idx]

        # train/test split on filtered-set indices (local)
        train_local, test_local = train_test_split(
            np.arange(len(y_filtered)),
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y_filtered if cfg.stratify else None,
        )

        # Convert to global indices (IMPORTANT)
        train_idx = keep_idx[train_local].astype(np.int64)
        test_idx = keep_idx[test_local].astype(np.int64)

        np.savez_compressed(path, train_idx=train_idx, test_idx=test_idx)
        return train_idx, test_idx, h

    # --------
    # Level B
    # --------
    def load_or_build_norm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        split_hash: str,
        cfg: NormConfig,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Returns payload + norm_hash
        payload: {X_train,X_test,y_train,y_test,scaler}
        """
        cfg_dict = asdict(cfg)
        key = {"cfg": cfg_dict, "split_hash": split_hash}
        h = config_hash(key)

        path = self.dir_B / f"norm_{h}.joblib"
        if path.exists():
            return joblib.load(path), h

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if cfg.method == "minmax":
            scaler = MinMaxScaler(feature_range=cfg.feature_range, clip=cfg.clip)
        elif cfg.method == "zscore":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Norm method không hợp lệ: {cfg.method}")

        X_train = scaler.fit_transform(X_train_raw).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test_raw).astype(np.float32, copy=False)

        payload = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
        }
        _atomic_joblib_dump(payload, path, compress=0)
        return payload, h

    # --------
    # Level E
    # --------
    def load_or_build_best_params(
        self,
        X_train: np.ndarray,
        norm_hash: str,
        cfg: OptimConfig,
        optimizer_fn: Callable[[np.ndarray, OptimConfig], Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        key = {"cfg": asdict(cfg), "norm_hash": norm_hash}
        h = config_hash(key)

        path = self.dir_E / f"optim_{h}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return float(d["theta"]), float(d["alpha"]), float(d["score"])

        theta, alpha, score = optimizer_fn(X_train, cfg)
        _atomic_write_text(
            path,
            json.dumps({"theta": theta, "alpha": alpha, "score": score}, ensure_ascii=False, indent=2),
        )
        return theta, alpha, score
