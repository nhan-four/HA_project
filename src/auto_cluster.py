"""
Auto Cluster - Tự động chạy và chọn số cụm tối ưu (2-9)

Features:
- Chạy song song nhiều số cụm
- Đánh giá và chọn cụm tốt nhất
- Tối ưu hóa với numpy vectorization
- Batch processing cho dataset lớn
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

from .config import MIN_CLUSTERS, MAX_CLUSTERS, MIN_VALUE, MAX_VALUE
from .logger import get_logger
from .clustering import HedgeAlgebraClustering, ClusteringResult, ParameterOptimizer
from .batch_processor import BatchProcessor, MemoryConfig
from .clustering_metrics import ClusteringEvaluator, ClusteringMetrics


@dataclass
class ClusterEvaluation:
    """
    Kết quả đánh giá của một cấu hình cụm.
    
    Attributes:
        n_clusters: Số cụm
        theta: Tham số theta
        alpha: Tham số alpha
        accuracy: Độ chính xác (nếu có labels)
        silhouette_score: Silhouette score
        total_distance: Tổng khoảng cách đến tâm cụm
        cluster_distribution: Phân bố samples trong các cụm
        training_time: Thời gian training (s)
        partition_coefficient: Partition Coefficient (PC)
        classification_entropy: Classification Entropy (CE)
        xie_beni_index: Xie-Beni Index (XB)
    """
    n_clusters: int
    theta: float
    alpha: float
    accuracy: float = 0.0
    silhouette_score: float = 0.0
    total_distance: float = 0.0
    cluster_distribution: Dict[int, int] = None
    training_time: float = 0.0
    partition_coefficient: float = 0.0
    classification_entropy: float = 0.0
    xie_beni_index: float = 0.0


@dataclass
class AutoClusterResult:
    """
    Kết quả của AutoCluster.
    
    Attributes:
        best_n_clusters: Số cụm tốt nhất
        best_evaluation: Đánh giá của cấu hình tốt nhất
        all_evaluations: Danh sách đánh giá của tất cả các cấu hình
        total_time: Tổng thời gian chạy
    """
    best_n_clusters: int
    best_evaluation: ClusterEvaluation
    all_evaluations: List[ClusterEvaluation]
    total_time: float
    
    def summary(self) -> str:
        """Tạo bảng tóm tắt kết quả."""
        lines = [
            "=" * 100,
            "📊 KẾT QUẢ AUTO CLUSTER",
            "=" * 100,
            "",
            f"🏆 Số cụm tốt nhất: {self.best_n_clusters}",
            f"   θ = {self.best_evaluation.theta:.4f}, α = {self.best_evaluation.alpha:.4f}",
            f"   Silhouette: {self.best_evaluation.silhouette_score:.4f}",
            f"   Total Distance: {self.best_evaluation.total_distance:.4f}",
            f"   PC: {self.best_evaluation.partition_coefficient:.4f}",
            f"   CE: {self.best_evaluation.classification_entropy:.4f}",
            f"   XB: {self.best_evaluation.xie_beni_index:.4f}",
            "",
            "📋 Chi tiết tất cả các cấu hình:",
            "-" * 100,
            f"{'N':>3} | {'Theta':>8} | {'Alpha':>8} | {'Silhouette':>12} | {'PC':>10} | {'CE':>10} | {'XB':>12} | {'Time(s)':>8}",
            "-" * 100
        ]
        
        for eval in sorted(self.all_evaluations, key=lambda x: x.n_clusters):
            lines.append(
                f"{eval.n_clusters:>3} | {eval.theta:>8.4f} | {eval.alpha:>8.4f} | "
                f"{eval.silhouette_score:>12.4f} | {eval.partition_coefficient:>10.4f} | "
                f"{eval.classification_entropy:>10.4f} | {eval.xie_beni_index:>12.4f} | "
                f"{eval.training_time:>8.2f}"
            )
        
        lines.extend([
            "-" * 100,
            f"⏱️ Tổng thời gian: {self.total_time:.2f}s",
            "=" * 100
        ])
        
        return "\n".join(lines)


class OptimizedClustering:
    """
    Phiên bản tối ưu hóa của HedgeAlgebraClustering.
    
    Sử dụng numpy vectorization để tăng tốc độ xử lý.
    """
    
    @staticmethod
    def calculate_semantic_values_vectorized(X: np.ndarray) -> np.ndarray:
        """Tính semantic values với vectorization (nhanh hơn)."""
        return np.mean(X, axis=1)
    
    @staticmethod
    def assign_to_clusters_vectorized(
        semantic_values: np.ndarray,
        cluster_centers: np.ndarray
    ) -> np.ndarray:
        """
        Gán điểm vào cụm với vectorization (nhanh hơn nhiều lần).
        
        Sử dụng broadcasting thay vì vòng lặp.
        """
        centers = np.array(cluster_centers)
        n_clusters = len(centers)
        
        # Tính boundaries giữa các cụm
        # boundary[i] = (centers[i] + centers[i+1]) / 2
        boundaries = (centers[:-1] + centers[1:]) / 2
        
        # Sử dụng searchsorted để tìm cụm nhanh
        # searchsorted trả về index nơi giá trị nên được chèn vào
        cluster_labels = np.searchsorted(boundaries, semantic_values, side='left') + 1

        return cluster_labels.astype(np.int32)
    
    @staticmethod
    def update_centers_vectorized(
        semantic_values: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        old_centers: np.ndarray
    ) -> np.ndarray:
        """Cập nhật tâm cụm. Nếu cụm rỗng -> giữ tâm cũ (tránh teleport)."""
        old_centers = np.asarray(old_centers, dtype=float)
        new_centers = np.empty(n_clusters, dtype=float)

        for i in range(n_clusters):
            cluster_id = i + 1
            mask = (cluster_labels == cluster_id)
            if np.any(mask):
                new_centers[i] = float(np.mean(semantic_values[mask]))
            else:
                new_centers[i] = float(old_centers[i])

        new_centers = np.clip(new_centers, MIN_VALUE, MAX_VALUE)
        return np.sort(new_centers)

    @staticmethod
    def calculate_total_distance_vectorized(
        semantic_values: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray
    ) -> float:
        """Tính tổng khoảng cách (L2 Squared) đồng bộ với core Ver 6.5."""
        centers = np.asarray(cluster_centers, dtype=float)
        label_indices = (cluster_labels - 1).astype(int)
        valid_mask = (label_indices >= 0) & (label_indices < len(centers))
        if not np.any(valid_mask):
            return 0.0
        sample_centers = centers[label_indices[valid_mask]]
        diffs = semantic_values[valid_mask] - sample_centers
        return float(np.sum(diffs ** 2))

    @staticmethod
    def calculate_silhouette_fast(
        semantic_values: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        sample_size: int = 5000
    ) -> float:
        """
        Tính Silhouette score nhanh (approximate).
        
        Sử dụng sampling nếu dataset lớn để tăng tốc.
        """
        n_samples = len(semantic_values)
        
        # Sampling nếu dataset lớn
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            values = semantic_values[indices]
            labels = cluster_labels[indices]
        else:
            values = semantic_values
            labels = cluster_labels
        
        n = len(values)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0.0
        
        # Tính silhouette cho mỗi sample
        silhouettes = np.zeros(n)
        
        for i in range(n):
            label_i = labels[i]
            
            # a(i): khoảng cách trung bình đến các điểm cùng cụm
            same_cluster = values[labels == label_i]
            if len(same_cluster) > 1:
                a_i = np.mean(np.abs(values[i] - same_cluster))
            else:
                a_i = 0
            
            # b(i): khoảng cách trung bình nhỏ nhất đến cụm khác
            b_i = np.inf
            for other_label in unique_labels:
                if other_label != label_i:
                    other_cluster = values[labels == other_label]
                    if len(other_cluster) > 0:
                        dist = np.mean(np.abs(values[i] - other_cluster))
                        b_i = min(b_i, dist)
            
            if b_i == np.inf:
                b_i = 0
            
            # Silhouette
            if max(a_i, b_i) > 0:
                silhouettes[i] = (b_i - a_i) / max(a_i, b_i)
        
        return np.mean(silhouettes)


class AutoClusterPipeline:
    """
    Pipeline tự động chạy và đánh giá nhiều số cụm.
    
    Chạy từ 2-9 cụm và chọn cấu hình tốt nhất dựa trên
    Silhouette score hoặc tổng khoảng cách.
    
    Example:
        >>> auto = AutoClusterPipeline(min_clusters=2, max_clusters=9)
        >>> result = auto.run(X)
        >>> print(result.summary())
        >>> print(f"Best: {result.best_n_clusters} clusters")
    """
    
    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 9,
        optimize_params: bool = True,
        max_memory_gb: float = 4.0,
        n_jobs: int = 1,
        log_level: str = "INFO",
        center_init: str = "ver6"
    ):
        """
        Khởi tạo AutoClusterPipeline.
        
        Args:
            min_clusters: Số cụm tối thiểu (>= 2)
            max_clusters: Số cụm tối đa (<= 10)
            optimize_params: Có tối ưu theta/alpha cho mỗi cấu hình không
            max_memory_gb: RAM tối đa (GB)
            n_jobs: Số jobs chạy song song (1 = sequential)
            log_level: Mức độ logging
        """
        self.min_clusters = max(MIN_CLUSTERS, min_clusters)
        self.max_clusters = min(MAX_CLUSTERS, max_clusters)
        self.optimize_params = optimize_params
        self.max_memory_gb = max_memory_gb
        self.n_jobs = n_jobs

        self.center_init = center_init
        if self.center_init not in ("ver6", "legacy"):
            raise ValueError(f"center_init phải là 'ver6' hoặc 'legacy', nhận được: {self.center_init}")
        self.logger = get_logger("AutoCluster", level=log_level, log_to_file=False)
        self.batch_processor = BatchProcessor(max_memory_gb=max_memory_gb, log_level="WARNING")
    
    def _evaluate_single_config(
        self,
        X: np.ndarray,
        n_clusters: int,
        theta: float = None,
        alpha: float = None
    ) -> ClusterEvaluation:
        """
        Đánh giá một cấu hình cụm.
        
        Args:
            X: Features
            n_clusters: Số cụm
            theta: Tham số theta (None = tối ưu)
            alpha: Tham số alpha (None = tối ưu)
        
        Returns:
            ClusterEvaluation: Kết quả đánh giá
        """
        start_time = time.time()
        
        # Tính semantic values
        semantic_values = OptimizedClustering.calculate_semantic_values_vectorized(X)
        
        # Tối ưu theta/alpha nếu cần
        if self.optimize_params and (theta is None or alpha is None):
            optimizer = ParameterOptimizer(
                theta_range=(0.1, 0.5, 0.05),
                alpha_range=(0.1, 0.5, 0.05),
                log_level="ERROR",
                center_init=self.center_init
            )
            theta, alpha, _ = optimizer.optimize(X, n_clusters)
        else:
            theta = theta or 0.5
            alpha = alpha or 0.5
        
        # Khởi tạo centers
        clustering = HedgeAlgebraClustering(
            n_clusters=n_clusters,
            theta=theta,
            alpha=alpha,
            log_level="ERROR",
            center_init=self.center_init
        )
        centers = np.array(clustering.initialize_cluster_centers())

        # Semantic Scaling (sync with core): map centers [0,1] into [min(Sd), max(Sd)]
        min_sd = float(np.min(semantic_values))
        max_sd = float(np.max(semantic_values))
        range_sd = max_sd - min_sd
        if range_sd > 1e-6:
            centers = min_sd + centers * range_sd
        
        # Clustering iterations
        for _ in range(50):  # Max iterations
            labels = OptimizedClustering.assign_to_clusters_vectorized(semantic_values, centers)
            new_centers = OptimizedClustering.update_centers_vectorized(semantic_values, labels, n_clusters, centers)
            
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers
        
        # Tính metrics
        total_distance = OptimizedClustering.calculate_total_distance_vectorized(
            semantic_values, labels, centers
        )
        silhouette = OptimizedClustering.calculate_silhouette_fast(
            semantic_values, labels, centers
        )
        
        # Tính clustering metrics (PC, CE, XB)
        try:
            evaluator = ClusteringEvaluator(log_level="ERROR")
            metrics = evaluator.evaluate(X, labels, centers, calculate_silhouette=False)
            pc = metrics.partition_coefficient
            ce = metrics.classification_entropy
            xb = metrics.xie_beni_index
        except Exception:
            pc = ce = xb = 0.0
        
        # Phân bố cụm
        distribution = {}
        for k in range(1, n_clusters + 1):
            distribution[k] = int(np.sum(labels == k))
        
        training_time = time.time() - start_time
        
        return ClusterEvaluation(
            n_clusters=n_clusters,
            theta=theta,
            alpha=alpha,
            silhouette_score=silhouette,
            total_distance=total_distance,
            cluster_distribution=distribution,
            training_time=training_time,
            partition_coefficient=pc,
            classification_entropy=ce,
            xie_beni_index=xb
        )
    
    def run(
        self,
        X: np.ndarray,
        selection_metric: str = "silhouette"
    ) -> AutoClusterResult:
        """
        Chạy auto clustering và chọn cấu hình tốt nhất.
        
        Args:
            X: Features array
            selection_metric: Metric để chọn cụm tốt nhất
                - "silhouette": Silhouette score cao nhất
                - "distance": Tổng khoảng cách thấp nhất
                - "elbow": Phương pháp elbow
        
        Returns:
            AutoClusterResult: Kết quả với cụm tốt nhất
        """
        start_time = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("🔄 BẮT ĐẦU AUTO CLUSTER")
        self.logger.info(f"   Số cụm: {self.min_clusters} → {self.max_clusters}")
        self.logger.info(f"   Tối ưu params: {self.optimize_params}")
        self.logger.info(f"   Selection metric: {selection_metric}")
        self.logger.info("=" * 70)
        
        evaluations = []
        cluster_range = range(self.min_clusters, self.max_clusters + 1)
        
        for n_clusters in cluster_range:
            self.logger.info(f"\n📍 Đang đánh giá {n_clusters} cụm...")
            
            eval_result = self._evaluate_single_config(X, n_clusters)
            evaluations.append(eval_result)
            
            self.logger.info(
                f"   ✅ Silhouette: {eval_result.silhouette_score:.4f}, "
                f"Distance: {eval_result.total_distance:.4f}, "
                f"Time: {eval_result.training_time:.2f}s"
            )
        
        # Chọn cấu hình tốt nhất
        if selection_metric == "silhouette":
            best_eval = max(evaluations, key=lambda x: x.silhouette_score)
        elif selection_metric == "distance":
            best_eval = min(evaluations, key=lambda x: x.total_distance)
        elif selection_metric == "elbow":
            best_eval = self._find_elbow(evaluations)
        else:
            best_eval = max(evaluations, key=lambda x: x.silhouette_score)
        
        total_time = time.time() - start_time
        
        result = AutoClusterResult(
            best_n_clusters=best_eval.n_clusters,
            best_evaluation=best_eval,
            all_evaluations=evaluations,
            total_time=total_time
        )
        
        self.logger.info("\n" + result.summary())
        
        return result
    
    def _find_elbow(self, evaluations: List[ClusterEvaluation]) -> ClusterEvaluation:
        """
        Tìm điểm elbow trong đồ thị distance.
        
        Sử dụng phương pháp tính góc để tìm điểm uốn.
        """
        n_points = len(evaluations)
        if n_points < 3:
            return evaluations[0]
        
        # Lấy distances
        distances = np.array([e.total_distance for e in evaluations])
        x = np.arange(n_points)
        
        # Normalize
        x_norm = x / x.max()
        d_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        
        # Tính khoảng cách từ mỗi điểm đến đường thẳng nối 2 đầu
        line_vec = np.array([x_norm[-1] - x_norm[0], d_norm[-1] - d_norm[0]])
        line_vec = line_vec / np.linalg.norm(line_vec)
        
        distances_to_line = []
        for i in range(n_points):
            point_vec = np.array([x_norm[i] - x_norm[0], d_norm[i] - d_norm[0]])
            # Khoảng cách vuông góc đến đường thẳng
            dist = np.abs(np.cross(line_vec, point_vec))
            distances_to_line.append(dist)
        
        elbow_idx = np.argmax(distances_to_line)
        return evaluations[elbow_idx]
    
    def run_with_batches(
        self,
        X: np.ndarray,
        batch_size: int = 50000,
        selection_metric: str = "silhouette"
    ) -> AutoClusterResult:
        """
        Chạy auto clustering với batch processing.
        
        Dùng cho dataset lớn để tránh tràn RAM.
        
        Args:
            X: Features array
            batch_size: Kích thước mỗi batch
            selection_metric: Metric để chọn cụm
        
        Returns:
            AutoClusterResult
        """
        n_samples = X.shape[0]
        
        if n_samples <= batch_size:
            return self.run(X, selection_metric)
        
        self.logger.info(f"📦 Dataset lớn ({n_samples:,} samples), sử dụng sampling")
        
        # Sample dữ liệu để tìm cấu hình tốt nhất
        sample_size = min(batch_size, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[indices]
        
        return self.run(X_sample, selection_metric)


def auto_cluster(
    X: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 9,
    optimize: bool = True,
    max_memory_gb: float = 4.0,
    center_init: str = "ver6"
) -> AutoClusterResult:
    """
    Hàm tiện ích để chạy auto clustering nhanh.
    
    Args:
        X: Features array
        min_clusters: Số cụm tối thiểu
        max_clusters: Số cụm tối đa
        optimize: Có tối ưu params không
        max_memory_gb: RAM tối đa
    
    Returns:
        AutoClusterResult
    
    Example:
        >>> result = auto_cluster(X, min_clusters=2, max_clusters=9)
        >>> print(f"Best: {result.best_n_clusters} clusters")
    """
    pipeline = AutoClusterPipeline(
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        optimize_params=optimize,
        max_memory_gb=max_memory_gb,
        log_level="INFO",
        center_init=center_init
    )
    return pipeline.run(X)

