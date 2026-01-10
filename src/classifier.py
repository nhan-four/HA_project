"""
Classifier - Training và Testing với ML models cho từng cụm

Module này chứa:
- ClusterClassifier: Class để train model cho từng cụm
- Các metrics đánh giá
- Hỗ trợ lưu/load models
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import joblib

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from .clustering import HedgeAlgebraClustering, ClusteringResult
from .logger import get_logger, TrainingLogger


@dataclass
class PredictionResult:
    """
    Kết quả dự đoán.
    
    Attributes:
        predictions: Nhãn dự đoán
        accuracy: Độ chính xác
        precision: Precision (macro average)
        recall: Recall (macro average)
        f1: F1-score (macro average)
        total_time: Tổng thời gian dự đoán
        average_time_per_sample: Thời gian trung bình mỗi sample
        classification_report: Báo cáo chi tiết
    """
    predictions: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float
    total_time: float
    average_time_per_sample: float
    classification_report: str


class ClusterClassifier:
    """
    Classifier cho Hedge Algebra Clustering.
    
    Train một ML model riêng cho mỗi cụm, sau đó sử dụng
    phân cụm để quyết định model nào sẽ được sử dụng để dự đoán.
    
    Attributes:
        clustering: HedgeAlgebraClustering instance
        base_classifier: ML model base (sklearn compatible)
        cluster_models: Dict chứa model của từng cụm
    
    Example:
        >>> classifier = ClusterClassifier(
        ...     n_clusters=3,
        ...     base_classifier=GradientBoostingClassifier()
        ... )
        >>> classifier.fit(X_train, y_train)
        >>> result = classifier.predict(X_test, y_test)
        >>> print(f"Accuracy: {result.accuracy:.4f}")
    """

    def __init__(
        self,
        n_clusters: int = 2,
        theta: float = 0.5,
        alpha: float = 0.5,
        base_classifier: Optional[BaseEstimator] = None,
        use_information_gain: bool = False,
        random_state: int = 42,
        log_level: str = "INFO",
        center_init: str = "ver6"
    ):
        """
        Khởi tạo ClusterClassifier.

        Args:
            n_clusters: Số cụm
            theta: Tham số theta cho ĐSGT
            alpha: Tham số alpha cho ĐSGT
            base_classifier: Model sklearn (mặc định GradientBoostingClassifier)
            use_information_gain: Có sử dụng IG weights không
            random_state: Seed
            log_level: Mức độ logging
            center_init: Chế độ khởi tạo tâm cụm ("ver6" hoặc "legacy")
        """
        if center_init not in ("ver6", "legacy"):
            raise ValueError(
                f"center_init phải là 'ver6' hoặc 'legacy', nhận được: {center_init}"
            )

        self.n_clusters = n_clusters
        self.theta = theta
        self.alpha = alpha
        self.use_information_gain = use_information_gain
        self.random_state = random_state
        self.center_init = center_init

        # Logger
        self.logger = get_logger("Classifier", level=log_level, log_to_file=False)
        self.training_logger = TrainingLogger(self.logger)

        # ML model
        if base_classifier is None:
            self.base_classifier = GradientBoostingClassifier(random_state=random_state)
        else:
            self.base_classifier = base_classifier

        # Clustering
        self.clustering = HedgeAlgebraClustering(
            n_clusters=n_clusters,
            theta=theta,
            alpha=alpha,
            log_level=log_level,
            center_init=center_init
        )

        # Models cho từng cụm
        self.cluster_models: Dict[int, BaseEstimator] = {}
        self.information_gain_weights: Optional[np.ndarray] = None

        # Kết quả training
        self._clustering_result: Optional[ClusteringResult] = None
        self._is_fitted: bool = False
        self._training_time: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        information_gain_weights: Optional[np.ndarray] = None
    ) -> 'ClusterClassifier':
        """
        Train classifier trên dữ liệu.
        
        Quy trình:
        1. Phân cụm dữ liệu training
        2. Train một model riêng cho mỗi cụm
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            information_gain_weights: Trọng số IG cho mỗi feature
        
        Returns:
            self: Để có thể chain methods
        
        Example:
            >>> classifier.fit(X_train, y_train).predict(X_test, y_test)
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("🚀 BẮT ĐẦU TRAINING")
        self.logger.info("=" * 60)
        
        # Lưu IG weights
        if information_gain_weights is not None:
            self.information_gain_weights = information_gain_weights
            self.use_information_gain = True
        
        # Log thông tin dữ liệu
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        self.training_logger.log_data_info(n_samples, n_features, n_classes)
        
        # 1. Phân cụm
        self.logger.info("\n📍 Bước 1: Phân cụm dữ liệu")
        
        # DEBUG: Kiểm tra X trước khi clustering
        print("🔍 DEBUG [Classifier] - NaN in X:", np.isnan(X).sum(), "Inf in X:", np.isinf(X).sum())
        print("🔍 DEBUG [Classifier] - X shape:", X.shape, "X min/max:", np.nanmin(X), np.nanmax(X))
        
        self._clustering_result = self.clustering.fit(
            X, 
            information_gain_weights=self.information_gain_weights
        )
        
        cluster_labels = self._clustering_result.cluster_labels
        
        # Log phân bố cụm
        cluster_distribution = {}
        for cluster_id in range(1, self.n_clusters + 1):
            count = int(np.sum(cluster_labels == cluster_id))
            cluster_distribution[cluster_id] = count
        self.training_logger.log_cluster_distribution(cluster_distribution)
        
        # 2. Train model cho từng cụm
        self.logger.info("\n📍 Bước 2: Training model cho từng cụm")
        
        # Áp dụng IG weights nếu có
        X_weighted = X
        if self.use_information_gain and self.information_gain_weights is not None:
            X_weighted = X * self.information_gain_weights
        
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_mask = (cluster_labels == cluster_id)
            X_cluster = X_weighted[cluster_mask]
            y_cluster = y[cluster_mask]
            
            n_cluster_samples = len(y_cluster)
            
            if n_cluster_samples == 0:
                self.logger.warning(f"   ⚠️ Cụm {cluster_id}: Không có dữ liệu!")
                continue
            
            unique_in_cluster = np.unique(y_cluster)
            
            if len(unique_in_cluster) < 2:
                self.logger.warning(
                    f"   ⚠️ Cụm {cluster_id}: Chỉ có 1 class ({unique_in_cluster[0]}), "
                    f"sử dụng constant predictor"
                )
                # Tạo một "model" đơn giản trả về class duy nhất
                from sklearn.dummy import DummyClassifier
                # Ép kiểu int nếu có thể, nếu không giữ nguyên
                constant_value = unique_in_cluster[0]
                if isinstance(constant_value, (np.floating, float)):
                    constant_value = int(constant_value)
                model = DummyClassifier(strategy='constant', constant=constant_value)
                model.fit(X_cluster, y_cluster)
                self.cluster_models[cluster_id] = model
                self.training_logger.log_training_result(cluster_id, success=True)
                continue
            
            try:
                # Clone model để mỗi cụm có model riêng
                from sklearn.base import clone
                model = clone(self.base_classifier)
                model.fit(X_cluster, y_cluster)
                self.cluster_models[cluster_id] = model
                
                self.training_logger.log_training_result(cluster_id, success=True)
                self.logger.debug(
                    f"      Samples: {n_cluster_samples}, Classes: {len(unique_in_cluster)}"
                )
                
            except Exception as e:
                # Fallback: dùng DummyClassifier
                self.logger.warning(f"   ⚠️ Cụm {cluster_id}: Fallback to DummyClassifier")
                from sklearn.dummy import DummyClassifier
                model = DummyClassifier(strategy='most_frequent')
                model.fit(X_cluster, y_cluster)
                self.cluster_models[cluster_id] = model
                self.training_logger.log_training_result(cluster_id, success=True)
        
        self._is_fitted = True
        self._training_time = time.time() - start_time
        
        self.logger.info(f"\n⏱️ Thời gian training: {self._training_time:.2f}s")
        self.logger.info("=" * 60)
        self.logger.info("✅ HOÀN TẤT TRAINING")
        self.logger.info("=" * 60)
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> PredictionResult:
        """
        Dự đoán trên dữ liệu mới.
        
        Quy trình:
        1. Xác định cụm của mỗi sample
        2. Sử dụng model của cụm đó để dự đoán
        
        Args:
            X: Features
            y_true: Labels thực (nếu có, để tính metrics)
        
        Returns:
            PredictionResult: Kết quả dự đoán và metrics
        
        Raises:
            ValueError: Nếu chưa fit
        """
        if not self._is_fitted:
            raise ValueError("Classifier chưa được fit. Gọi fit() trước.")
        
        self.logger.info("=" * 60)
        self.logger.info("🔮 BẮT ĐẦU DỰ ĐOÁN")
        self.logger.info("=" * 60)
        
        n_samples = X.shape[0]
        self.logger.info(f"   📊 Số samples: {n_samples:,}")
        
        # Áp dụng IG weights
        X_weighted = X
        if self.use_information_gain and self.information_gain_weights is not None:
            X_weighted = X * self.information_gain_weights
        
        # FIX: Label có thể là string (vd: 'DDoS-ICMP_Flood'), nên không được dùng float array
        # Dùng dtype theo y_true nếu có, fallback object
        pred_dtype = y_true.dtype if y_true is not None else object
        predictions = np.empty(n_samples, dtype=pred_dtype)
        prediction_times = []
        
        # Dự đoán từng sample
        for sample_index in range(n_samples):
            start_time = time.time()
            
            sample = X_weighted[sample_index:sample_index+1]
            
            # Xác định cụm
            cluster_label = self.clustering.predict(sample)[0]
            cluster_id = int(cluster_label)
            
            # Dự đoán bằng model của cụm
            if cluster_id in self.cluster_models:
                prediction = self.cluster_models[cluster_id].predict(sample)[0]
            else:
                # Fallback: dùng model của cụm đầu tiên có sẵn
                first_available_cluster = min(self.cluster_models.keys())
                prediction = self.cluster_models[first_available_cluster].predict(sample)[0]
                self.logger.debug(
                    f"   ⚠️ Sample {sample_index}: Cụm {cluster_id} không có model, "
                    f"dùng cụm {first_available_cluster}"
                )
            
            predictions[sample_index] = prediction
            prediction_times.append(time.time() - start_time)
        
        total_time = sum(prediction_times)
        average_time = np.mean(prediction_times)
        
        # Tính metrics nếu có y_true
        if y_true is not None:
            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average='macro', zero_division=0)
            recall = recall_score(y_true, predictions, average='macro', zero_division=0)
            f1 = f1_score(y_true, predictions, average='macro', zero_division=0)
            report = classification_report(y_true, predictions, digits=4, zero_division=0)
            
            self.training_logger.log_metrics({
                'Accuracy': accuracy,
                'Precision (macro)': precision,
                'Recall (macro)': recall,
                'F1-score (macro)': f1
            })
            
            self.training_logger.log_summary(
                train_time=self._training_time,
                test_time=total_time,
                accuracy=accuracy
            )
        else:
            accuracy = precision = recall = f1 = 0.0
            report = "Không có y_true để tính metrics"
        
        self.logger.info("=" * 60)
        self.logger.info("✅ HOÀN TẤT DỰ ĐOÁN")
        self.logger.info("=" * 60)
        
        return PredictionResult(
            predictions=predictions,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            total_time=total_time,
            average_time_per_sample=average_time,
            classification_report=report
        )
    
    def save_models(self, directory: str = "models"):
        """
        Lưu các models đã train.
        
        Args:
            directory: Thư mục lưu models
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for cluster_id, model in self.cluster_models.items():
            model_path = directory / f"cluster_{cluster_id}_model.pkl"
            joblib.dump(model, model_path)
            self.logger.info(f"   💾 Đã lưu model cụm {cluster_id}: {model_path}")
        
        # Lưu metadata
        metadata = {
            'n_clusters': self.n_clusters,
            'theta': self.theta,
            'alpha': self.alpha,
            'cluster_centers': self.clustering.cluster_centers,
            'information_gain_weights': self.information_gain_weights
        }
        metadata_path = directory / "metadata.pkl"
        joblib.dump(metadata, metadata_path)
        self.logger.info(f"   💾 Đã lưu metadata: {metadata_path}")
    
    def load_models(self, directory: str = "models"):
        """
        Load các models đã lưu.
        
        Args:
            directory: Thư mục chứa models
        """
        directory = Path(directory)
        
        # Load metadata
        metadata_path = directory / "metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        self.n_clusters = metadata['n_clusters']
        self.theta = metadata['theta']
        self.alpha = metadata['alpha']
        self.information_gain_weights = metadata['information_gain_weights']
        self.clustering._cluster_centers = metadata['cluster_centers']
        self.clustering._is_fitted = True
        
        # Load models
        for cluster_id in range(1, self.n_clusters + 1):
            model_path = directory / f"cluster_{cluster_id}_model.pkl"
            if model_path.exists():
                self.cluster_models[cluster_id] = joblib.load(model_path)
                self.logger.info(f"   📂 Đã load model cụm {cluster_id}")
        
        self._is_fitted = True
        self.logger.info("   ✅ Load models thành công")
    
    @property
    def training_time(self) -> float:
        """Trả về thời gian training."""
        return self._training_time
    
    @property
    def cluster_centers(self) -> Optional[List[float]]:
        """Trả về tâm cụm."""
        return self.clustering.cluster_centers

