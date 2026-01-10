"""
Pipeline - Pipeline hoàn chỉnh từ dữ liệu thô đến kết quả

Đây là module chính để sử dụng, gói gọn toàn bộ quy trình:
1. Load dữ liệu (CSV/NPY)
2. Tiền xử lý (normalize, feature selection)
3. Phân cụm (Hedge Algebra)
4. Training model cho từng cụm
5. Testing và đánh giá

Example:
    >>> from src import HedgeAlgebraPipeline
    >>> 
    >>> # Cách 1: Sử dụng với file CSV
    >>> pipeline = HedgeAlgebraPipeline(n_clusters=3)
    >>> result = pipeline.run("data.csv")
    >>> print(f"Accuracy: {result.accuracy:.4f}")
    >>> 
    >>> # Cách 2: Sử dụng với dữ liệu đã có
    >>> pipeline = HedgeAlgebraPipeline(n_clusters=3)
    >>> result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import ClusteringConfig
from .logger import get_logger
from .data_loader import DataLoader
from .clustering import HedgeAlgebraClustering, ParameterOptimizer
from .classifier import ClusterClassifier, PredictionResult
from .batch_processor import BatchProcessor, MemoryConfig
from .auto_cluster import AutoClusterPipeline, AutoClusterResult
from .cache_utils import CacheManager, CleanConfig, SplitConfig, NormConfig, OptimConfig


@dataclass
class PipelineResult:
    """
    Kết quả của toàn bộ pipeline.
    
    Attributes:
        accuracy: Độ chính xác trên test set
        precision: Precision (macro)
        recall: Recall (macro)
        f1: F1-score (macro)
        training_time: Thời gian training (giây)
        testing_time: Thời gian testing (giây)
        n_train_samples: Số samples training
        n_test_samples: Số samples testing
        n_features: Số features
        n_classes: Số classes
        n_clusters: Số cụm
        cluster_centers: Tâm các cụm
        cluster_distribution: Phân bố samples trong các cụm
        classification_report: Báo cáo phân loại chi tiết
        theta: Tham số theta (có thể đã optimize)
        alpha: Tham số alpha (có thể đã optimize)
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_time: float
    testing_time: float
    n_train_samples: int
    n_test_samples: int
    n_features: int
    n_classes: int
    n_clusters: int
    cluster_centers: list
    cluster_distribution: dict
    classification_report: str
    theta: float
    alpha: float
    
    def summary(self) -> str:
        """Trả về tóm tắt kết quả."""
        lines = [
            "=" * 60,
            "📋 TÓM TẮT KẾT QUẢ PIPELINE",
            "=" * 60,
            f"",
            f"📊 DỮ LIỆU:",
            f"   • Train samples: {self.n_train_samples:,}",
            f"   • Test samples: {self.n_test_samples:,}",
            f"   • Features: {self.n_features:,}",
            f"   • Classes: {self.n_classes}",
            f"",
            f"🔧 CẤU HÌNH:",
            f"   • Số cụm: {self.n_clusters}",
            f"   • Theta: {self.theta:.4f}",
            f"   • Alpha: {self.alpha:.4f}",
            f"   • Tâm cụm: {[f'{c:.4f}' for c in self.cluster_centers]}",
            f"",
            f"📈 KẾT QUẢ:",
            f"   • Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)",
            f"   • Precision: {self.precision:.4f}",
            f"   • Recall: {self.recall:.4f}",
            f"   • F1-score: {self.f1:.4f}",
            f"",
            f"⏱️ THỜI GIAN:",
            f"   • Training: {self.training_time:.2f}s",
            f"   • Testing: {self.testing_time:.4f}s",
            f"",
            "=" * 60
        ]
        return "\n".join(lines)


class HedgeAlgebraPipeline:
    """
    Pipeline hoàn chỉnh cho Hedge Algebra Clustering.
    
    Đây là class chính để sử dụng module. Hỗ trợ:
    - Load dữ liệu từ CSV hoặc NPY
    - Tự động tiền xử lý
    - Tối ưu hóa tham số (optional)
    - Training và testing
    - Logging chi tiết
    
    Attributes:
        config: Cấu hình pipeline
        data_loader: DataLoader instance
        classifier: ClusterClassifier instance
    
    Example:
        >>> # Sử dụng cơ bản
        >>> pipeline = HedgeAlgebraPipeline(n_clusters=3)
        >>> result = pipeline.run("data.csv")
        >>> print(result.summary())
        
        >>> # Sử dụng với cấu hình tùy chỉnh
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> pipeline = HedgeAlgebraPipeline(
        ...     n_clusters=4,
        ...     theta=0.3,
        ...     alpha=0.4,
        ...     classifier=RandomForestClassifier(n_estimators=100),
        ...     use_information_gain=True,
        ...     optimize_parameters=True
        ... )
        >>> result = pipeline.run("data.csv", label_column="target")
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        theta: float = 0.5,
        alpha: float = 0.5,
        classifier: Optional[BaseEstimator] = None,
        use_information_gain: bool = False,
        optimize_parameters: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        log_level: str = 'INFO',
        log_to_file: bool = True,
        log_dir: str = 'logs',
        center_init: str = 'ver6',
        use_cache: bool = False,
        cache_dir: str = 'cache',
        clean_version: int = 1,
        min_per_class: int = 5
    ):
        """
        Khởi tạo HedgeAlgebraPipeline.
        
        Args:
            n_clusters: Số cụm (2-10)
            theta: Tham số theta (0 < theta < 1)
            alpha: Tham số alpha (0 < alpha < 1)
            classifier: ML model sklearn (mặc định GradientBoostingClassifier)
            use_information_gain: Có sử dụng IG weights không
            optimize_parameters: Có tối ưu theta/alpha không
            test_size: Tỷ lệ test set (0 < test_size < 1)
            random_state: Seed cho reproducibility
            log_level: Mức độ logging (DEBUG, INFO, WARNING, ERROR)
            log_to_file: Có ghi log ra file không
            log_dir: Thư mục chứa file log
        """
        self.n_clusters = n_clusters
        self.theta = theta
        self.alpha = alpha
        self.use_information_gain = use_information_gain
        self.optimize_parameters = optimize_parameters
        self.test_size = test_size
        self.random_state = random_state
        self.center_init = center_init
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.clean_version = clean_version
        self.min_per_class = min_per_class
        
        # Classifier
        if classifier is None:
            self.base_classifier = GradientBoostingClassifier(
                random_state=random_state
            )
        else:
            self.base_classifier = classifier
        
        # Logger
        self.logger = get_logger(
            "Pipeline",
            level=log_level,
            log_to_file=log_to_file,
            log_dir=log_dir
        )
        
        # Components (sẽ được khởi tạo khi chạy)
        self.data_loader: Optional[DataLoader] = None
        self.classifier: Optional[ClusterClassifier] = None
        self._result: Optional[PipelineResult] = None
    
    def run(
        self,
        file_path: str,
        label_column: Optional[str] = None,
        normalize_method: str = "minmax",
        target_names: Optional[list] = None
    ) -> PipelineResult:
        """
        Chạy pipeline với file dữ liệu.
        
        Đây là method chính để sử dụng pipeline.
        
        Args:
            file_path: Đường dẫn file CSV hoặc NPY
            label_column: Tên cột label (cho CSV, mặc định "label")
            normalize_method: Phương pháp chuẩn hóa ("minmax" hoặc "zscore"/"standard")
            target_names: Tên các class (optional, để hiển thị)
        
        Returns:
            PipelineResult: Kết quả của pipeline
        
        Example:
            >>> pipeline = HedgeAlgebraPipeline(n_clusters=3)
            >>> result = pipeline.run("data.csv")
            >>> print(f"Accuracy: {result.accuracy:.4f}")
        """
        self.logger.info("=" * 70)
        self.logger.info("🚀 HEDGE ALGEBRA CLUSTERING PIPELINE")
        self.logger.info(f"   File: {file_path}")
        self.logger.info(f"   Số cụm: {self.n_clusters}")
        self.logger.info(f"   Sử dụng IG: {self.use_information_gain}")
        self.logger.info(f"   Tối ưu tham số: {self.optimize_parameters}")
        self.logger.info(f"   Sử dụng cache: {self.use_cache}")
        self.logger.info("=" * 70)
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
        # Set default label_column
        if label_column is None:
            label_column = "label"
        
        if self.use_cache:
            # --- CACHE FLOW (Level A -> D -> B) ---
            cm = CacheManager(file_path, cache_root=self.cache_dir)
            
            # Level A: Clean
            c_cfg = CleanConfig(
                label_column=label_column,
                clean_version=self.clean_version,
                nan_policy="fill0"
            )
            X_clean, y_raw, clean_h = cm.load_or_build_clean(c_cfg)
            self.logger.info(f"✅ Level A (Clean): {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
            
            # Level D: Split (Fair Comparison Key)
            s_cfg = SplitConfig(
                test_size=self.test_size,
                random_state=self.random_state,
                min_per_class=self.min_per_class
            )
            train_idx, test_idx, split_h = cm.load_or_build_split(y_raw, clean_h, s_cfg)
            self.logger.info(f"✅ Level D (Split): {len(train_idx)} train, {len(test_idx)} test")
            
            # Level B: Normalize
            # Map method name từ pipeline sang cache config
            norm_method_map = "zscore" if normalize_method in ("zscore", "standard") else "minmax"
            n_cfg = NormConfig(method=norm_method_map)
            
            # Hàm load_or_build_norm trả về (payload, norm_h)
            data_norm, norm_h = cm.load_or_build_norm(X_clean, y_raw, train_idx, test_idx, split_h, n_cfg)
            
            X_train, X_test = data_norm["X_train"], data_norm["X_test"]
            y_train, y_test = data_norm["y_train"], data_norm["y_test"]
            
            self.logger.info(f"✅ Level B (Normalize): Loaded from cache. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            # Tính IG weights nếu cần (chưa có trong cache, tính lại)
            ig_weights = None
            if self.use_information_gain:
                self.data_loader = DataLoader(log_level="INFO")
                ig_weights = self.data_loader.calculate_information_gain_ratio(X_train, y_train)
                self.logger.info("✅ Information Gain Ratio calculated")
        else:
            # Fallback: Logic cũ không dùng cache
            self.data_loader = DataLoader(log_level="INFO")
            
            X_train, X_test, y_train, y_test, ig_weights = self.data_loader.load_and_preprocess(
                file_path=file_path,
                label_column=label_column,
                normalize_method=normalize_method,
                remove_constant=True,
                calculate_ig=self.use_information_gain,
                test_size=self.test_size,
                random_state=self.random_state
            )
            norm_h = None  # Không có hash khi không dùng cache
        
        # --- Level E Hook (Optimization) ---
        if self.optimize_parameters:
            if self.use_cache:
                o_cfg = OptimConfig(
                    center_init=self.center_init,
                    n_clusters=self.n_clusters,
                    theta_range=(0.01, 0.5, 0.01),
                    alpha_range=(0.01, 0.5, 0.01)
                )
                
                # Wrapper function để gọi optimizer cũ
                def _run_optim(X, cfg):
                    opt = ParameterOptimizer(
                        center_init=cfg.center_init,
                        theta_range=cfg.theta_range,
                        alpha_range=cfg.alpha_range,
                        log_level=self.log_level
                    )
                    # ParameterOptimizer.optimize trả về (theta, alpha, min_distance)
                    return opt.optimize(X, cfg.n_clusters)
                
                self.theta, self.alpha, best_score = cm.load_or_build_best_params(
                    X_train, norm_h, o_cfg, _run_optim
                )
                self.logger.info(f"✅ Level E (Optimization): theta={self.theta:.4f}, alpha={self.alpha:.4f} (Score: {best_score:.4f})")
            else:
                # Logic cũ: tối ưu trực tiếp
                self.logger.info("\n📍 Tối ưu hóa tham số theta và alpha")
                optimizer = ParameterOptimizer(log_level="INFO", center_init=self.center_init)
                self.theta, self.alpha, min_distance = optimizer.optimize(X_train, self.n_clusters)
                self.logger.info(f"   ✅ Tối ưu: θ={self.theta:.4f}, α={self.alpha:.4f}, Distance={min_distance:.4f}")
        
        # Chạy với dữ liệu đã load
        return self.run_with_data(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            information_gain_weights=ig_weights
        )
    
    def run_with_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        information_gain_weights: Optional[np.ndarray] = None
    ) -> PipelineResult:
        """
        Chạy pipeline với dữ liệu đã có sẵn.
        
        Sử dụng khi đã có dữ liệu train/test sẵn.
        
        Args:
            X_train: Features training
            X_test: Features testing
            y_train: Labels training
            y_test: Labels testing
            information_gain_weights: Trọng số IG (optional)
        
        Returns:
            PipelineResult: Kết quả của pipeline
        
        Example:
            >>> pipeline = HedgeAlgebraPipeline(n_clusters=3)
            >>> result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        """
        pipeline_start_time = time.time()
        
        n_train_samples, n_features = X_train.shape
        n_test_samples = X_test.shape[0]
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        theta = self.theta
        alpha = self.alpha
        
        # 2. Tối ưu tham số (nếu cần)
        if self.optimize_parameters:
            self.logger.info("\n📍 Tối ưu hóa tham số theta và alpha")
            optimizer = ParameterOptimizer(log_level="INFO", center_init=self.center_init)
            theta, alpha, min_distance = optimizer.optimize(
                X_train, self.n_clusters
            )
            self.theta = theta
            self.alpha = alpha
        
        # 3. Khởi tạo và train classifier
        self.classifier = ClusterClassifier(
            n_clusters=self.n_clusters,
            theta=theta,
            alpha=alpha,
            base_classifier=self.base_classifier,
            use_information_gain=self.use_information_gain,
            random_state=self.random_state,
            log_level="INFO",
            center_init=self.center_init
        )
        
        self.classifier.fit(
            X_train, y_train,
            information_gain_weights=information_gain_weights
        )
        
        training_time = self.classifier.training_time
        
        # 4. Test và đánh giá
        prediction_result = self.classifier.predict(X_test, y_test)
        
        # 5. Tính phân bố cụm
        cluster_distribution = {}
        clustering_result = self.classifier._clustering_result
        if clustering_result is not None:
            for cluster_id in range(1, self.n_clusters + 1):
                count = int(np.sum(clustering_result.cluster_labels == cluster_id))
                cluster_distribution[cluster_id] = count
        
        # 6. Tạo kết quả
        self._result = PipelineResult(
            accuracy=prediction_result.accuracy,
            precision=prediction_result.precision,
            recall=prediction_result.recall,
            f1=prediction_result.f1,
            training_time=training_time,
            testing_time=prediction_result.total_time,
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters=self.n_clusters,
            cluster_centers=self.classifier.cluster_centers,
            cluster_distribution=cluster_distribution,
            classification_report=prediction_result.classification_report,
            theta=theta,
            alpha=alpha
        )
        
        total_time = time.time() - pipeline_start_time
        
        # Log summary
        self.logger.info("\n" + self._result.summary())
        self.logger.info(f"\n⏱️ Tổng thời gian pipeline: {total_time:.2f}s")
        
        return self._result
    
    def get_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Lấy predictions cho dữ liệu mới (không có labels).
        
        Args:
            X: Features
        
        Returns:
            np.ndarray: Predictions
        """
        if self.classifier is None or not self.classifier._is_fitted:
            raise ValueError("Pipeline chưa được chạy. Gọi run() trước.")
        
        result = self.classifier.predict(X)
        return result.predictions
    
    def save(self, directory: str = "saved_pipeline"):
        """
        Lưu pipeline đã train.
        
        Args:
            directory: Thư mục lưu
        """
        if self.classifier is None:
            raise ValueError("Pipeline chưa được chạy.")
        
        self.logger.info(f"\n💾 Lưu pipeline vào: {directory}")
        self.classifier.save_models(directory)
    
    def load(self, directory: str = "saved_pipeline"):
        """
        Load pipeline đã lưu.
        
        Args:
            directory: Thư mục chứa pipeline
        """
        self.logger.info(f"\n📂 Load pipeline từ: {directory}")
        
        self.classifier = ClusterClassifier(log_level="INFO", center_init=self.center_init)
        self.classifier.load_models(directory)
        
        # Cập nhật config từ metadata
        self.n_clusters = self.classifier.n_clusters
        self.theta = self.classifier.theta
        self.alpha = self.classifier.alpha
    
    @property
    def result(self) -> Optional[PipelineResult]:
        """Trả về kết quả pipeline (nếu đã chạy)."""
        return self._result
    
    def run_auto_cluster(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 9,
        selection_metric: str = "silhouette",
        information_gain_weights: Optional[np.ndarray] = None
    ) -> Tuple['PipelineResult', 'AutoClusterResult']:
        """
        Tự động chạy từ min_clusters đến max_clusters và chọn cụm tốt nhất.
        
        Args:
            X_train: Features training
            X_test: Features testing
            y_train: Labels training
            y_test: Labels testing
            min_clusters: Số cụm tối thiểu (mặc định 2)
            max_clusters: Số cụm tối đa (mặc định 9)
            selection_metric: Metric để chọn cụm ("silhouette", "distance", "elbow")
            information_gain_weights: Trọng số IG (optional)
        
        Returns:
            Tuple[PipelineResult, AutoClusterResult]: Kết quả pipeline và auto cluster
        
        Example:
            >>> pipeline = HedgeAlgebraPipeline()
            >>> result, auto_result = pipeline.run_auto_cluster(
            ...     X_train, X_test, y_train, y_test,
            ...     min_clusters=2, max_clusters=9
            ... )
            >>> print(f"Best clusters: {auto_result.best_n_clusters}")
            >>> print(f"Accuracy: {result.accuracy:.4f}")
        """
        self.logger.info("=" * 70)
        self.logger.info("🔄 AUTO CLUSTER PIPELINE")
        self.logger.info(f"   Tìm số cụm tốt nhất từ {min_clusters} đến {max_clusters}")
        self.logger.info("=" * 70)
        
        # Bước 1: Chạy auto cluster để tìm số cụm tốt nhất
        auto_pipeline = AutoClusterPipeline(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            optimize_params=self.optimize_parameters,
            log_level="INFO",
            center_init=self.center_init
        )
        
        auto_result = auto_pipeline.run(X_train, selection_metric=selection_metric)
        
        # Bước 2: Cập nhật config với cụm tốt nhất
        best_n_clusters = auto_result.best_n_clusters
        best_theta = auto_result.best_evaluation.theta
        best_alpha = auto_result.best_evaluation.alpha
        
        self.n_clusters = best_n_clusters
        self.theta = best_theta
        self.alpha = best_alpha
        
        self.logger.info(f"\n🏆 Số cụm tốt nhất: {best_n_clusters}")
        self.logger.info(f"   θ = {best_theta:.4f}, α = {best_alpha:.4f}")
        
        # Bước 3: Chạy pipeline với cấu hình tốt nhất
        self.logger.info("\n📍 Training với cấu hình tối ưu...")
        
        result = self.run_with_data(
            X_train, X_test, y_train, y_test,
            information_gain_weights=information_gain_weights
        )
        
        return result, auto_result


def quick_run(
    file_path: str,
    n_clusters: int = 2,
    label_column: Optional[str] = None,
    optimize: bool = False
) -> PipelineResult:
    """
    Hàm tiện ích để chạy nhanh pipeline.
    
    Đây là cách đơn giản nhất để sử dụng module.
    
    Args:
        file_path: Đường dẫn file CSV hoặc NPY
        n_clusters: Số cụm
        label_column: Tên cột label (optional)
        optimize: Có tối ưu tham số không
    
    Returns:
        PipelineResult: Kết quả
    
    Example:
        >>> from src.pipeline import quick_run
        >>> result = quick_run("data.csv", n_clusters=3)
        >>> print(f"Accuracy: {result.accuracy:.4f}")
    """
    pipeline = HedgeAlgebraPipeline(
        n_clusters=n_clusters,
        optimize_parameters=optimize
    )
    return pipeline.run(file_path, label_column=label_column)


def quick_auto_run(
    file_path: str,
    min_clusters: int = 2,
    max_clusters: int = 9,
    label_column: Optional[str] = None,
    max_memory_gb: float = 4.0
) -> Tuple[PipelineResult, AutoClusterResult]:
    """
    Hàm tiện ích để chạy auto cluster nhanh.
    
    Tự động tìm số cụm tốt nhất từ min_clusters đến max_clusters.
    
    Args:
        file_path: Đường dẫn file CSV hoặc NPY
        min_clusters: Số cụm tối thiểu
        max_clusters: Số cụm tối đa
        label_column: Tên cột label (optional)
        max_memory_gb: RAM tối đa (GB)
    
    Returns:
        Tuple[PipelineResult, AutoClusterResult]
    
    Example:
        >>> from src.pipeline import quick_auto_run
        >>> result, auto_result = quick_auto_run("data.csv", min_clusters=2, max_clusters=9)
        >>> print(f"Best: {auto_result.best_n_clusters} clusters")
        >>> print(f"Accuracy: {result.accuracy:.4f}")
    """
    # Load dữ liệu
    data_loader = DataLoader(log_level="INFO")
    X_train, X_test, y_train, y_test, ig_weights = data_loader.load_and_preprocess(
        file_path=file_path,
        label_column=label_column,
        normalize_method="minmax",
        calculate_ig=False
    )
    
    # Chạy auto cluster
    pipeline = HedgeAlgebraPipeline(
        n_clusters=2,  # Sẽ được cập nhật bởi auto cluster
        log_level="INFO",
        log_to_file=False
    )
    
    return pipeline.run_auto_cluster(
        X_train, X_test, y_train, y_test,
        min_clusters=min_clusters,
        max_clusters=max_clusters
    )


class LargeDatasetPipeline:
    """
    Pipeline cho dataset lớn (hàng triệu dòng).
    
    Sử dụng batch processing để tránh tràn RAM.
    
    Example:
        >>> pipeline = LargeDatasetPipeline(max_memory_gb=8.0)
        >>> result = pipeline.run("large_data.csv", n_clusters=5)
    """
    
    def __init__(
        self,
        max_memory_gb: float = 4.0,
        batch_size: Optional[int] = None,
        log_level: str = "INFO"
    ):
        """
        Khởi tạo LargeDatasetPipeline.
        
        Args:
            max_memory_gb: RAM tối đa được sử dụng (GB)
            batch_size: Kích thước batch (None = tự động)
            log_level: Mức độ logging
        """
        self.max_memory_gb = max_memory_gb
        self.batch_size = batch_size
        self.logger = get_logger("LargeDataPipeline", level=log_level, log_to_file=False)
        self.batch_processor = BatchProcessor(
            max_memory_gb=max_memory_gb,
            batch_size=batch_size,
            log_level=log_level
        )
    
    def run(
        self,
        file_path: str,
        n_clusters: int = 2,
        label_column: Optional[str] = None,
        sample_for_training: int = 100000
    ) -> PipelineResult:
        """
        Chạy pipeline cho dataset lớn.
        
        Sử dụng sampling cho training và batch processing cho prediction.
        
        Args:
            file_path: Đường dẫn file
            n_clusters: Số cụm
            label_column: Tên cột label
            sample_for_training: Số samples dùng để train
        
        Returns:
            PipelineResult
        """
        self.logger.info("=" * 70)
        self.logger.info("🚀 LARGE DATASET PIPELINE")
        self.logger.info(f"   Max RAM: {self.max_memory_gb} GB")
        self.logger.info(f"   Sample for training: {sample_for_training:,}")
        self.logger.info("=" * 70)
        
        # Load và sample dữ liệu
        data_loader = DataLoader(log_level="INFO")
        X_train, X_test, y_train, y_test, _ = data_loader.load_and_preprocess(
            file_path=file_path,
            label_column=label_column,
            normalize_method="minmax"
        )
        
        # Sample nếu dataset quá lớn
        n_train = X_train.shape[0]
        if n_train > sample_for_training:
            self.logger.info(f"📦 Sampling {sample_for_training:,} từ {n_train:,} samples")
            indices = np.random.choice(n_train, sample_for_training, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        # Chạy pipeline
        pipeline = HedgeAlgebraPipeline(
            n_clusters=n_clusters,
            log_level="INFO",
            log_to_file=False
        )
        
        return pipeline.run_with_data(
            X_train_sample, X_test,
            y_train_sample, y_test
        )

