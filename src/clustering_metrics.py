"""
Clustering Evaluation Metrics - Các chỉ số đánh giá chất lượng phân cụm

Bao gồm:
- Partition Coefficient (PC): Đo lường mức độ phân biệt và độ tập trung của các cụm mờ
- Classification Entropy (CE): Đo lường mức độ không chắc chắn trong việc phân cụm
- Xie-Beni Index (XB): Đo lường chất lượng phân cụm dựa trên tỷ lệ compactness/separation
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .logger import get_logger


@dataclass
class ClusteringMetrics:
    """
    Kết quả các chỉ số đánh giá phân cụm.
    
    Attributes:
        partition_coefficient: Partition Coefficient (PC) - càng cao càng tốt (0-1)
        classification_entropy: Classification Entropy (CE) - càng thấp càng tốt (0-1)
        xie_beni_index: Xie-Beni Index (XB) - càng thấp càng tốt (>0)
        silhouette_score: Silhouette Score (nếu có)
    """
    partition_coefficient: float
    classification_entropy: float
    xie_beni_index: float
    silhouette_score: float = 0.0
    
    def summary(self) -> str:
        """Tạo bảng tóm tắt metrics."""
        lines = [
            "=" * 60,
            "📊 CLUSTERING EVALUATION METRICS",
            "=" * 60,
            f"",
            f"Partition Coefficient (PC):     {self.partition_coefficient:.6f} (↑ cao hơn = tốt hơn)",
            f"Classification Entropy (CE):     {self.classification_entropy:.6f} (↓ thấp hơn = tốt hơn)",
            f"Xie-Beni Index (XB):            {self.xie_beni_index:.6f} (↓ thấp hơn = tốt hơn)",
            f"Silhouette Score:                {self.silhouette_score:.6f} (↑ cao hơn = tốt hơn)",
            f"",
            "=" * 60
        ]
        return "\n".join(lines)


class ClusteringEvaluator:
    """
    Class để tính các chỉ số đánh giá phân cụm.
    
    Hỗ trợ đánh giá chất lượng phân cụm dựa trên:
    - Membership matrix (ma trận độ thuộc)
    - Cluster centers (tâm cụm)
    - Data points (điểm dữ liệu)
    
    Example:
        >>> evaluator = ClusteringEvaluator()
        >>> metrics = evaluator.evaluate(X, cluster_labels, cluster_centers)
        >>> print(metrics.summary())
    """
    
    def __init__(self, log_level: str = "WARNING"):
        """
        Khởi tạo ClusteringEvaluator.
        
        Args:
            log_level: Mức độ logging
        """
        self.logger = get_logger("ClusteringEvaluator", level=log_level, log_to_file=False)
    
    def calculate_membership_matrix(
        self,
        semantic_values: np.ndarray,
        cluster_centers: np.ndarray,
        fuzziness: float = 2.0
    ) -> np.ndarray:
        """
        Tính ma trận membership (độ thuộc) cho fuzzy clustering.
        
        Membership được tính dựa trên khoảng cách từ điểm đến tâm cụm.
        Sử dụng công thức tương tự Fuzzy C-Means.
        
        Args:
            semantic_values: Giá trị ngữ nghĩa của các điểm (n_samples,)
            cluster_centers: Tâm các cụm (n_clusters,) - có thể là list hoặc array
            fuzziness: Tham số mờ (m > 1), mặc định 2.0
        
        Returns:
            np.ndarray: Ma trận membership (n_clusters, n_samples)
                       Mỗi cột tổng = 1 (probabilistic)
        """
        # Convert cluster_centers sang numpy array nếu là list
        cluster_centers = np.array(cluster_centers)
        semantic_values = np.array(semantic_values)
        
        n_samples = len(semantic_values)
        n_clusters = len(cluster_centers)
        
        # Tính khoảng cách từ mỗi điểm đến mỗi tâm cụm
        distances = np.abs(
            semantic_values[:, np.newaxis] - cluster_centers[np.newaxis, :]
        )
        
        # Tránh chia cho 0
        distances = np.maximum(distances, 1e-10)
        
        # Tính membership theo công thức Fuzzy C-Means
        # u_ij = 1 / sum_k (d_ij / d_kj)^(2/(m-1))
        power = 2.0 / (fuzziness - 1.0)
        membership = np.zeros((n_clusters, n_samples))
        
        for i in range(n_clusters):
            for j in range(n_samples):
                ratio = distances[j, i] / distances[j, :]
                membership[i, j] = 1.0 / np.sum(ratio ** power)
        
        # Chuẩn hóa để đảm bảo tổng = 1
        membership = membership / np.sum(membership, axis=0, keepdims=True)
        
        return membership
    
    def calculate_partition_coefficient(
        self,
        membership_matrix: np.ndarray
    ) -> float:
        """
        Tính Partition Coefficient (PC).
        
        PC đo lường mức độ phân biệt và độ tập trung của các cụm mờ.
        Giá trị PC nằm trong khoảng [1/n_clusters, 1]:
        - PC = 1/n_clusters: Các cụm hoàn toàn không phân biệt (worst)
        - PC = 1: Các cụm hoàn toàn phân biệt (best)
        
        Công thức: PC = (1/n) * sum_i sum_j (u_ij)^2
        
        Args:
            membership_matrix: Ma trận membership (n_clusters, n_samples)
        
        Returns:
            float: Partition Coefficient (0-1)
        """
        n_samples = membership_matrix.shape[1]
        
        # PC = (1/n) * sum_i sum_j (u_ij)^2
        pc = np.mean(np.sum(membership_matrix ** 2, axis=0))
        
        return float(pc)
    
    def calculate_classification_entropy(
        self,
        membership_matrix: np.ndarray
    ) -> float:
        """
        Tính Classification Entropy (CE).
        
        CE đo lường mức độ không chắc chắn trong việc phân cụm.
        Giá trị CE nằm trong khoảng [0, log(n_clusters)]:
        - CE = 0: Phân cụm hoàn toàn chắc chắn (best)
        - CE = log(n_clusters): Phân cụm hoàn toàn không chắc chắn (worst)
        
        Công thức: CE = -(1/n) * sum_i sum_j (u_ij * log(u_ij))
        
        Args:
            membership_matrix: Ma trận membership (n_clusters, n_samples)
        
        Returns:
            float: Classification Entropy (0-1, normalized)
        """
        n_samples = membership_matrix.shape[1]
        n_clusters = membership_matrix.shape[0]
        
        # Tránh log(0)
        membership_safe = np.maximum(membership_matrix, 1e-10)
        
        # CE = -(1/n) * sum_i sum_j (u_ij * log(u_ij))
        ce = -np.mean(np.sum(membership_safe * np.log(membership_safe), axis=0))
        
        # Normalize về [0, 1]
        max_ce = np.log(n_clusters)
        if max_ce > 0:
            ce_normalized = ce / max_ce
        else:
            ce_normalized = 0.0
        
        return float(ce_normalized)
    
    def calculate_xie_beni_index(
        self,
        X: np.ndarray,
        membership_matrix: np.ndarray,
        cluster_centers: np.ndarray
    ) -> float:
        """
        Tính Xie-Beni Index (XB).
        
        XB đo lường chất lượng phân cụm dựa trên tỷ lệ:
        - Compactness: Độ chặt trong từng cụm (numerator)
        - Separation: Khoảng cách giữa các cụm (denominator)
        
        XB = (sum_i sum_j (u_ij^2 * ||x_j - v_i||^2)) / (n * min_{i!=k} ||v_i - v_k||^2)
        
        Giá trị XB càng thấp càng tốt:
        - XB thấp: Các cụm chặt chẽ và tách biệt rõ ràng
        
        Args:
            X: Features array (n_samples, n_features)
            membership_matrix: Ma trận membership (n_clusters, n_samples)
            cluster_centers: Tâm cụm (n_clusters,) - semantic values của centers
        
        Returns:
            float: Xie-Beni Index (>0)
        """
        n_samples = X.shape[0]
        n_clusters = len(cluster_centers)
        
        # Tính semantic values của X
        semantic_values = np.mean(X, axis=1)
        
        # Tử số: Tổng compactness
        # sum_i sum_j (u_ij^2 * ||x_j - v_i||^2)
        numerator = 0.0
        
        for i in range(n_clusters):
            for j in range(n_samples):
                distance = abs(semantic_values[j] - cluster_centers[i])
                numerator += (membership_matrix[i, j] ** 2) * (distance ** 2)
        
        # Mẫu số: Separation (khoảng cách nhỏ nhất giữa các tâm cụm)
        min_center_distance = np.inf
        
        for i in range(n_clusters):
            for k in range(n_clusters):
                if i != k:
                    distance = abs(cluster_centers[i] - cluster_centers[k])
                    if distance < min_center_distance:
                        min_center_distance = distance
        
        # Tránh chia cho 0
        if min_center_distance < 1e-10:
            min_center_distance = 1e-10
        
        denominator = n_samples * (min_center_distance ** 2)
        
        # XB Index
        xb = numerator / denominator
        
        return float(xb)
    
    def calculate_silhouette_score(
        self,
        semantic_values: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        sample_size: int = 5000
    ) -> float:
        """
        Tính Silhouette Score (approximate).
        
        Sử dụng sampling nếu dataset lớn để tăng tốc.
        
        Args:
            semantic_values: Giá trị ngữ nghĩa
            cluster_labels: Nhãn cụm
            cluster_centers: Tâm cụm
            sample_size: Số samples để tính (nếu dataset lớn)
        
        Returns:
            float: Silhouette Score (-1 đến 1)
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
        
        return float(np.mean(silhouettes))
    
    def evaluate(
        self,
        X: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        calculate_silhouette: bool = True
    ) -> ClusteringMetrics:
        """
        Tính tất cả các chỉ số đánh giá phân cụm.
        
        Đây là method chính để sử dụng.
        
        Args:
            X: Features array (n_samples, n_features)
            cluster_labels: Nhãn cụm (n_samples,) - 1-indexed
            cluster_centers: Tâm cụm (n_clusters,) - semantic values (có thể là list)
            calculate_silhouette: Có tính Silhouette Score không
        
        Returns:
            ClusteringMetrics: Kết quả các chỉ số
        
        Example:
            >>> evaluator = ClusteringEvaluator()
            >>> metrics = evaluator.evaluate(X, labels, centers)
            >>> print(metrics.summary())
        """
        # Convert sang numpy arrays
        X = np.array(X)
        cluster_labels = np.array(cluster_labels)
        cluster_centers = np.array(cluster_centers)
        
        # Tính semantic values
        semantic_values = np.mean(X, axis=1)
        
        # Tính membership matrix
        membership_matrix = self.calculate_membership_matrix(
            semantic_values, cluster_centers
        )
        
        # Tính các metrics
        pc = self.calculate_partition_coefficient(membership_matrix)
        ce = self.calculate_classification_entropy(membership_matrix)
        xb = self.calculate_xie_beni_index(X, membership_matrix, cluster_centers)
        
        # Silhouette (optional)
        silhouette = 0.0
        if calculate_silhouette:
            silhouette = self.calculate_silhouette_score(
                semantic_values, cluster_labels, cluster_centers
            )
        
        return ClusteringMetrics(
            partition_coefficient=pc,
            classification_entropy=ce,
            xie_beni_index=xb,
            silhouette_score=silhouette
        )


def quick_evaluate(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray
) -> ClusteringMetrics:
    """
    Hàm tiện ích để đánh giá nhanh.
    
    Args:
        X: Features array
        cluster_labels: Nhãn cụm
        cluster_centers: Tâm cụm
    
    Returns:
        ClusteringMetrics
    
    Example:
        >>> metrics = quick_evaluate(X, labels, centers)
        >>> print(f"PC: {metrics.partition_coefficient:.4f}")
    """
    evaluator = ClusteringEvaluator(log_level="ERROR")
    return evaluator.evaluate(X, cluster_labels, cluster_centers)

