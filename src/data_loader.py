"""
Data Loader - Load và tiền xử lý dữ liệu

Hỗ trợ:
- Load từ CSV, NPY
- Chuẩn hóa dữ liệu (Min-Max, Z-Score)
- Feature selection (loại bỏ cột constant)
- Tính Information Gain Ratio
- Split train/test
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .logger import get_logger


class DataLoader:
    """
    Class để load và tiền xử lý dữ liệu cho Hedge Algebra Clustering.
    
    Attributes:
        logger: Logger instance
        scaler: Scaler đã được fit (nếu có)
        removed_columns: Danh sách các cột đã bị loại bỏ
        information_gain_ratio: IG ratio của các features
    
    Example:
        >>> loader = DataLoader()
        >>> X_train, X_test, y_train, y_test = loader.load_and_split("data.csv")
        >>> print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Khởi tạo DataLoader.
        
        Args:
            log_level: Mức độ logging
        """
        self.logger = get_logger("DataLoader", level=log_level, log_to_file=False)
        self.scaler = None
        self.removed_columns = []
        self.information_gain_ratio = None
        self._feature_names = None
    
    def load_csv(
        self,
        file_path: str,
        label_column: Optional[str] = None,
        label_column_index: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dữ liệu từ file CSV.
        
        Args:
            file_path: Đường dẫn tới file CSV
            label_column: Tên cột label (nếu biết)
            label_column_index: Index cột label (-1 = cột cuối)
        
        Returns:
            Tuple[X, y]: Features và labels
        
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu dữ liệu không hợp lệ
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
        self.logger.info(f"📂 Đang load file: {file_path.name}")
        
        # Load CSV
        dataframe = pd.read_csv(file_path)
        self._feature_names = list(dataframe.columns)
        
        # Xác định cột label
        if label_column is not None:
            if label_column not in dataframe.columns:
                raise ValueError(f"Không tìm thấy cột '{label_column}' trong dữ liệu")
            y = dataframe[label_column].values
            X = dataframe.drop(columns=[label_column]).values
            self._feature_names.remove(label_column)
        else:
            # Mặc định: cột cuối là label
            y = dataframe.iloc[:, label_column_index].values
            X = dataframe.iloc[:, :label_column_index].values if label_column_index == -1 else \
                np.delete(dataframe.values, label_column_index, axis=1)
            self._feature_names = self._feature_names[:label_column_index] if label_column_index == -1 else \
                self._feature_names[:label_column_index] + self._feature_names[label_column_index+1:]
        
        self._log_data_info(X, y)
        return X.astype(np.float64), y
    
    def load_numpy(
        self,
        file_path: str,
        label_column_index: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dữ liệu từ file NPY.
        
        Args:
            file_path: Đường dẫn tới file NPY
            label_column_index: Index cột label (-1 = cột cuối)
        
        Returns:
            Tuple[X, y]: Features và labels
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
        self.logger.info(f"📂 Đang load file: {file_path.name}")
        
        data = np.load(file_path, allow_pickle=True)
        
        if label_column_index == -1:
            X = data[:, :-1]
            y = data[:, -1]
        else:
            y = data[:, label_column_index]
            X = np.delete(data, label_column_index, axis=1)
        
        self._log_data_info(X, y)
        return X.astype(np.float64), y
    
    def _log_data_info(self, X: np.ndarray, y: np.ndarray):
        """Ghi log thông tin dữ liệu."""
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        self.logger.info(f"  📊 Số samples: {n_samples:,}")
        self.logger.info(f"  📊 Số features: {n_features:,}")
        self.logger.info(f"  📊 Số classes: {n_classes} {list(unique_classes)[:5]}{'...' if n_classes > 5 else ''}")
    
    def normalize(
        self,
        X: np.ndarray,
        method: str = "minmax",
        fit: bool = True
    ) -> np.ndarray:
        """
        Chuẩn hóa dữ liệu.
        
        Args:
            X: Dữ liệu cần chuẩn hóa
            method: Phương pháp chuẩn hóa ("minmax" hoặc "zscore")
            fit: True nếu cần fit scaler, False nếu dùng scaler đã fit
        
        Returns:
            np.ndarray: Dữ liệu đã chuẩn hóa
        """
        self.logger.info(f"🔧 Chuẩn hóa dữ liệu: {method}")
        
        if fit:
            if method == "minmax":
                self.scaler = MinMaxScaler()
            elif method == "zscore":
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"Phương pháp không hợp lệ: {method}")
            
            X_normalized = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler chưa được fit. Gọi normalize() với fit=True trước.")
            X_normalized = self.scaler.transform(X)
        
        return X_normalized
    
    def remove_constant_columns(
        self,
        X: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Loại bỏ các cột chỉ có 1 giá trị (constant columns).
        
        Args:
            X: Dữ liệu đầu vào
            fit: True nếu cần xác định các cột cần loại, False nếu dùng list đã xác định
        
        Returns:
            np.ndarray: Dữ liệu sau khi loại bỏ cột constant
        """
        if fit:
            self.removed_columns = []
            for column_index in range(X.shape[1]):
                unique_values = np.unique(X[:, column_index])
                if len(unique_values) == 1:
                    self.removed_columns.append(column_index)
            
            if self.removed_columns:
                self.logger.info(f"🗑️  Loại bỏ {len(self.removed_columns)} cột constant: {self.removed_columns}")
        
        if self.removed_columns:
            X = np.delete(X, self.removed_columns, axis=1)
        
        return X
    
    def calculate_information_gain_ratio(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Tính Information Gain Ratio cho các features.
        
        Information Gain Ratio = Information Gain / Entropy của feature
        Giá trị cao hơn = feature quan trọng hơn
        
        Args:
            X: Features
            y: Labels
        
        Returns:
            np.ndarray: IG ratio cho mỗi feature
        """
        self.logger.info("📊 Tính Information Gain Ratio...")
        
        n_samples, n_features = X.shape
        
        # Tính entropy của dataset
        dataset_entropy = self._calculate_entropy(y)
        
        information_gain_list = []
        entropy_feature_list = []
        
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            
            # Tính conditional entropy H(Y|X)
            conditional_entropy = 0
            feature_entropy = 0
            
            for value in unique_values:
                mask = feature_values == value
                subset_labels = y[mask]
                weight = np.sum(mask) / n_samples
                
                # H(Y|X=value)
                conditional_entropy += weight * self._calculate_entropy(subset_labels)
                
                # H(X)
                if weight > 0:
                    feature_entropy -= weight * np.log2(weight)
            
            # Information Gain = H(Y) - H(Y|X)
            information_gain = dataset_entropy - conditional_entropy
            information_gain_list.append(information_gain)
            entropy_feature_list.append(feature_entropy)
        
        # Information Gain Ratio = IG / H(X)
        entropy_feature_array = np.array(entropy_feature_list)
        entropy_feature_array[entropy_feature_array == 0] = 1e-10  # Tránh chia cho 0
        
        self.information_gain_ratio = np.array(information_gain_list) / entropy_feature_array
        
        self.logger.info(f"  📈 IG Ratio range: [{self.information_gain_ratio.min():.4f}, {self.information_gain_ratio.max():.4f}]")
        
        return self.information_gain_ratio
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Tính entropy của một tập labels."""
        if len(y) == 0:
            return 0
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        
        # H(Y) = -Σ p(y) * log2(p(y))
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def apply_information_gain_weights(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Áp dụng IG ratio làm trọng số cho features.
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Features đã được weight
        """
        if self.information_gain_ratio is None:
            raise ValueError("Chưa tính IG ratio. Gọi calculate_information_gain_ratio() trước.")
        
        if X.shape[1] != len(self.information_gain_ratio):
            raise ValueError(
                f"Số features ({X.shape[1]}) không khớp với số IG ratio ({len(self.information_gain_ratio)})"
            )
        
        return X * self.information_gain_ratio
    
    def split_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Chia dữ liệu thành train và test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Tỷ lệ test set
            random_state: Seed
            stratify: Có giữ tỷ lệ classes không
        
        Returns:
            Tuple[X_train, X_test, y_train, y_test]
        """
        self.logger.info(f"✂️  Chia dữ liệu: train={1-test_size:.0%}, test={test_size:.0%}")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        self.logger.info(f"  📊 Train: {X_train.shape[0]:,} samples")
        self.logger.info(f"  📊 Test: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def load_and_preprocess(
        self,
        file_path: str,
        label_column: Optional[str] = None,
        normalize_method: str = "minmax",
        remove_constant: bool = True,
        calculate_ig: bool = False,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Pipeline đầy đủ: Load → Preprocess → Split.
        
        Đây là method chính để sử dụng DataLoader.
        
        Args:
            file_path: Đường dẫn tới file dữ liệu (CSV hoặc NPY)
            label_column: Tên cột label (cho CSV)
            normalize_method: Phương pháp chuẩn hóa
            remove_constant: Có loại bỏ cột constant không
            calculate_ig: Có tính IG ratio không
            test_size: Tỷ lệ test set
            random_state: Seed
        
        Returns:
            Tuple[X_train, X_test, y_train, y_test, ig_ratio]
        
        Example:
            >>> loader = DataLoader()
            >>> X_train, X_test, y_train, y_test, ig = loader.load_and_preprocess(
            ...     "data.csv",
            ...     normalize_method="minmax",
            ...     calculate_ig=True
            ... )
        """
        self.logger.info("=" * 60)
        self.logger.info("🚀 BẮT ĐẦU LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU")
        self.logger.info("=" * 60)
        
        # 1. Load dữ liệu
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            X, y = self.load_csv(str(file_path), label_column=label_column)
        elif file_path.suffix.lower() == '.npy':
            X, y = self.load_numpy(str(file_path))
        else:
            raise ValueError(f"Không hỗ trợ định dạng: {file_path.suffix}")
        
        # 2. Loại bỏ cột constant
        if remove_constant:
            X = self.remove_constant_columns(X)
        
        # 3. Chuẩn hóa
        X = self.normalize(X, method=normalize_method)
        
        # 4. Tính IG ratio (nếu cần)
        ig_ratio = None
        if calculate_ig:
            ig_ratio = self.calculate_information_gain_ratio(X, y)
        
        # 5. Split
        X_train, X_test, y_train, y_test = self.split_train_test(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.logger.info("=" * 60)
        self.logger.info("✅ HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU")
        self.logger.info("=" * 60)
        
        return X_train, X_test, y_train, y_test, ig_ratio
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        """Trả về tên các features (nếu có)."""
        return self._feature_names

