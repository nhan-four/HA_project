"""
Clustering - Logic phân cụm dựa trên Đại số gia tử (Hedge Algebra)

Phiên bản: Ver 6.5 (Production Final)
Thay đổi:
1. [Feature] Dual Init Mode ("ver6" & "legacy").
2. [Safety] Validate fail-fast cho center_init trong cả Clustering và Optimizer.
3. [Safety] Fallback an toàn, giữ tâm cũ khi cụm rỗng.
4. [Optimization] Full-fit optimizer loop & L2 Squared Loss.
5. [Polish] Cleanup unused imports & Refined warning messages.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# [CLEANUP] Bỏ MIN_CLUSTERS vì đã validate cứng n >= 2
from .config import MIN_VALUE, MAX_VALUE, MAX_CLUSTERS
from .logger import get_logger
from .clustering_metrics import ClusteringEvaluator, ClusteringMetrics


@dataclass
class ClusteringResult:
    """
    Kết quả của quá trình phân cụm.
    """
    cluster_labels: np.ndarray
    cluster_centers: List[float]
    n_iterations: int
    converged: bool
    semantic_values: np.ndarray
    metrics: Optional['ClusteringMetrics'] = None


class HedgeAlgebraClustering:
    """
    Phân cụm dựa trên Đại số gia tử (Hedge Algebra).
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        theta: float = 0.5,
        alpha: float = 0.5,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        log_level: str = "INFO",
        center_init: str = "ver6"
    ):
        self.logger = get_logger("Clustering", level=log_level, log_to_file=False)
        
        self._validate_parameters(n_clusters, theta, alpha, center_init)
        
        self.n_clusters = n_clusters
        self.theta = theta
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.center_init = center_init
        
        self._cluster_centers: Optional[List[float]] = None
        self._is_fitted: bool = False
    
    def _validate_parameters(self, n_clusters: int, theta: float, alpha: float, center_init: str):
        """Validate tham số đầu vào."""
        if n_clusters < 2:
             raise ValueError(f"n_clusters phải >= 2, nhận được: {n_clusters}")
        
        # [POLISH] Warning trung tính hơn
        if n_clusters > MAX_CLUSTERS:
            self.logger.warning(
                f"⚠️ n_clusters={n_clusters} > MAX_CLUSTERS={MAX_CLUSTERS}. "
                f"Sẽ dùng công thức mở rộng/heuristic; nếu không có công thức sẽ fallback linspace."
            )

        if not 0 < theta < 1:
            raise ValueError(f"theta phải trong khoảng (0, 1)")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha phải trong khoảng (0, 1)")
            
        if center_init not in ("ver6", "legacy"):
            raise ValueError(f"center_init phải là 'ver6' hoặc 'legacy', nhận được: {center_init}")
    
    def initialize_cluster_centers(
        self,
        n_clusters: Optional[int] = None,
        theta: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> List[float]:
        """Wrapper khởi tạo tâm cụm."""
        mode = getattr(self, "center_init", "ver6")
        
        if mode == "legacy":
            return self._initialize_cluster_centers_legacy(n_clusters, theta, alpha)
        return self._initialize_cluster_centers_ver6(n_clusters, theta, alpha)

    def _initialize_cluster_centers_ver6(
        self,
        n_clusters: Optional[int] = None,
        theta: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> List[float]:
        """Logic Ver 6.3 (Heuristic ngữ nghĩa)."""
        n = n_clusters or self.n_clusters
        t = theta or self.theta
        a = alpha or self.alpha
        
        val_theta = t
        val_c_neg = t * (1 - a)                  # Thấp
        val_c_pos = t + a * (1 - t)              # Cao

        val_lc_neg = t * (1 - a * 0.5)           # Little -
        val_vc_neg = t * (1 - a) * (1 - a)       # Very -
        
        val_lc_pos = t + a * 0.5 * (1 - t)       # Little +
        val_vc_pos = t + a * (1 - t) * (1 + a)   # Very +

        formulas = {
            2: [val_c_neg, val_c_pos],
            3: [val_c_neg, val_theta, val_c_pos],
            4: [val_vc_neg, val_lc_neg, val_lc_pos, val_vc_pos],
            5: [val_vc_neg, val_lc_neg, val_theta, val_lc_pos, val_vc_pos],
            6: [val_vc_neg, val_c_neg, val_lc_neg, val_lc_pos, val_c_pos, val_vc_pos],
            7: [val_vc_neg, val_c_neg, val_lc_neg, val_theta, val_lc_pos, val_c_pos, val_vc_pos],
            8: np.linspace(val_vc_neg, val_vc_pos, 8).tolist(),
            9: np.linspace(val_vc_neg, val_vc_pos, 9).tolist(),
            10: np.linspace(val_vc_neg, val_vc_pos, 10).tolist(),
        }
        
        centers = formulas.get(n)
        if centers is None:
            centers = np.linspace(val_vc_neg, val_vc_pos, n).tolist()
            
        centers_arr = np.asarray(centers, dtype=float)
        centers_arr = np.clip(centers_arr, MIN_VALUE, MAX_VALUE)
        
        return sorted(centers_arr.tolist())

    def _initialize_cluster_centers_legacy(
        self,
        n_clusters: Optional[int] = None,
        theta: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> List[float]:
        """Legacy Init (Code cũ của project)."""
        n = n_clusters or self.n_clusters
        t = theta or self.theta
        a = alpha or self.alpha

        legacy_formulas = {
            2: [t * (1 - a), t + a * (1 - t)],
            3: [t * (1 - a)**2, t, t + a * (1 - t) * (1 - a)],
            4: [t * (1 - a), t * (1 - a / 2), t + a / 2 * (1 - t), t + a * (1 - t)],
            5: [t * (1 - a), t * (1 - a / 2), t, t + a / 2 * (1 - t), t + a * (1 - t)],
            6: [t * (1 - a), t * (1 - a / 2), t * (1 - a / 4), t + a / 4 * (1 - t), t + a / 2 * (1 - t), t + a * (1 - t)],
            7: [t * (1 - a), t * (1 - a / 2), t * (1 - a / 4), t, t + a / 4 * (1 - t), t + a / 2 * (1 - t), t + a * (1 - t)],
            8: [t * (1 - a), t * (1 - 3*a / 4), t * (1 - a / 2), t * (1 - a / 4), t + a / 4 * (1 - t), t + a / 2 * (1 - t), t + 3*a / 4 * (1 - t), t + a * (1 - t)],
            9: [t * (1 - a), t * (1 - 3*a / 4), t * (1 - a / 2), t * (1 - a / 4), t, t + a / 4 * (1 - t), t + a / 2 * (1 - t), t + 3*a / 4 * (1 - t), t + a * (1 - t)],
            10: [t * (1 - a), t * (1 - 4*a / 5), t * (1 - 3*a / 5), t * (1 - 2*a / 5), t * (1 - a / 5), t + a / 5 * (1 - t), t + 2*a / 5 * (1 - t), t + 3*a / 5 * (1 - t), t + 4*a / 5 * (1 - t), t + a * (1 - t)],
        }

        centers = legacy_formulas.get(n)
        if centers is None:
            centers = np.linspace(t * (1 - a), t + a * (1 - t), n).tolist()

        centers_arr = np.asarray(centers, dtype=float)
        centers_arr = np.clip(centers_arr, MIN_VALUE, MAX_VALUE)
        return sorted(centers_arr.tolist())

    def calculate_semantic_values(self, X: np.ndarray) -> np.ndarray:
        """Tính giá trị ngữ nghĩa trung bình."""
        return np.mean(X, axis=1)
    
    def assign_to_clusters(
        self,
        semantic_values: np.ndarray,
        cluster_centers: List[float]
    ) -> np.ndarray:
        """Gán nhãn cụm (Midpoint thresholding)."""
        n_clusters = len(cluster_centers)
        
        boundaries = []
        for i in range(n_clusters - 1):
            mid = (cluster_centers[i] + cluster_centers[i+1]) / 2
            boundaries.append(mid)
        
        boundaries_arr = np.asarray(boundaries, dtype=float)
        cluster_labels = np.searchsorted(boundaries_arr, semantic_values, side='left') + 1
        
        return cluster_labels.astype(np.int32)
    
    def update_cluster_centers(
        self,
        semantic_values: np.ndarray,
        cluster_labels: np.ndarray,
        old_centers: List[float]
    ) -> List[float]:
        """Cập nhật tâm cụm (Giữ tâm cũ nếu cụm rỗng)."""
        new_centers = []
        n_clusters = len(old_centers)
        
        for cluster_id in range(1, n_clusters + 1):
            mask = (cluster_labels == cluster_id)
            if np.any(mask):
                new_center = float(np.mean(semantic_values[mask]))
            else:
                new_center = float(old_centers[cluster_id - 1])
            new_centers.append(float(np.clip(new_center, MIN_VALUE, MAX_VALUE)))
        
        return sorted(new_centers)
    
    def check_convergence(self, old_centers: List[float], new_centers: List[float]) -> bool:
        for old, new in zip(old_centers, new_centers):
            if abs(old - new) >= self.tolerance:
                return False
        return True
    
    def fit(
        self,
        X: np.ndarray,
        information_gain_weights: Optional[np.ndarray] = None
    ) -> ClusteringResult:
        """Thực hiện phân cụm."""
        self.logger.info(f"🔄 Bắt đầu phân cụm ({self.center_init.upper()}) với {self.n_clusters} cụm")
        self.logger.info(f"   θ={self.theta:.4f}, α={self.alpha:.4f}")
        
        if information_gain_weights is not None:
            X = X * information_gain_weights
        
        # DEBUG: Kiểm tra NaN/Inf trong X trước khi tính semantic
        print("🔍 DEBUG - NaN in X:", np.isnan(X).sum(), "Inf in X:", np.isinf(X).sum())
        
        semantic_values = self.calculate_semantic_values(X)
        
        # DEBUG: Kiểm tra NaN/Inf trong semantic_values và unique values
        print("🔍 DEBUG - NaN in semantic:", np.isnan(semantic_values).sum(), "Inf in semantic:", np.isinf(semantic_values).sum())
        print("🔍 DEBUG - semantic min/max:", np.nanmin(semantic_values), np.nanmax(semantic_values))
        print("🔍 DEBUG - semantic unique approx:", np.unique(np.round(semantic_values, 6))[:20])
        
        min_sd = np.min(semantic_values)
        max_sd = np.max(semantic_values)
        range_sd = max_sd - min_sd
        self.logger.debug(f"   Dải ngữ nghĩa thực tế: [{min_sd:.4f}, {max_sd:.4f}]")
        
        # Gọi wrapper init
        raw_centers = self.initialize_cluster_centers()
        
        # DEBUG: In raw_centers
        print("🔍 DEBUG - raw_centers:", raw_centers[:10] if len(raw_centers) >= 10 else raw_centers)
        
        # Feature: Semantic Scaling
        if range_sd > 1e-6:
            cluster_centers = [min_sd + c * range_sd for c in raw_centers]
        else:
            cluster_centers = raw_centers
        
        # DEBUG: In scaled centers và boundary
        print("🔍 DEBUG - scaled_centers:", cluster_centers[:10] if len(cluster_centers) >= 10 else cluster_centers)
        if len(cluster_centers) >= 2:
            boundary_last = (cluster_centers[-2] + cluster_centers[-1]) / 2
            print("🔍 DEBUG - boundary last:", boundary_last)
        print("🔍 DEBUG - semantic min/max:", semantic_values.min(), semantic_values.max())
            
        self.logger.debug(f"   Initial centers: {[f'{c:.4f}' for c in cluster_centers]}")
        
        converged = False
        cluster_labels = np.zeros(len(semantic_values), dtype=np.int32)
        iteration = -1
        
        for iteration in range(self.max_iterations):
            cluster_labels = self.assign_to_clusters(semantic_values, cluster_centers)
            new_centers = self.update_cluster_centers(
                semantic_values, cluster_labels, cluster_centers
            )
            
            converged = self.check_convergence(cluster_centers, new_centers)
            cluster_centers = new_centers
            
            if converged:
                self.logger.info(f"   ✅ Hội tụ tại iteration {iteration + 1}")
                break
            
        if not converged and self.max_iterations > 0:
            self.logger.warning(f"   ⚠️ Không hội tụ sau {self.max_iterations} iterations")
        
        self._cluster_centers = cluster_centers
        self._is_fitted = True
        
        metrics = None
        try:
            evaluator = ClusteringEvaluator(log_level="ERROR")
            metrics = evaluator.evaluate(X, cluster_labels, cluster_centers)
        except Exception:
            pass
        
        return ClusteringResult(
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            n_iterations=(iteration + 1) if iteration >= 0 else 0,
            converged=converged,
            semantic_values=semantic_values,
            metrics=metrics
        )
    
    def predict(
        self,
        X: np.ndarray,
        information_gain_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Chưa fit. Gọi fit() trước.")
        
        if information_gain_weights is not None:
            X = X * information_gain_weights
        
        semantic_values = self.calculate_semantic_values(X)
        return self.assign_to_clusters(semantic_values, self._cluster_centers)
    
    @property
    def cluster_centers(self) -> Optional[List[float]]:
        return self._cluster_centers


class ParameterOptimizer:
    """
    Tối ưu hóa tham số theta và alpha.
    """
    
    def __init__(
        self,
        theta_range: Tuple[float, float, float] = (0.01, 0.5, 0.01),
        alpha_range: Tuple[float, float, float] = (0.01, 0.5, 0.01),
        log_level: str = "INFO",
        center_init: str = "ver6"
    ):
        # [FAIL-FAST] Validate ngay khi khởi tạo
        if center_init not in ("ver6", "legacy"):
            raise ValueError(f"center_init phải là 'ver6' hoặc 'legacy', nhận được: {center_init}")

        self.theta_values = np.arange(*theta_range)
        self.alpha_values = np.arange(*alpha_range)
        self.logger = get_logger("Optimizer", level=log_level, log_to_file=False)
        self.center_init = center_init
    
    def calculate_total_distance(
        self,
        semantic_values: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: List[float]
    ) -> float:
        """Tính tổng bình phương khoảng cách ngữ nghĩa (L2 Squared)."""
        total_distance = 0.0
        for cluster_id, center in enumerate(cluster_centers, start=1):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_points = semantic_values[cluster_mask]
            if len(cluster_points) > 0:
                squared_diff = (cluster_points - center) ** 2
                total_distance += np.sum(squared_diff)
        return total_distance
    
    def optimize(
        self,
        X: np.ndarray,
        n_clusters: int
    ) -> Tuple[float, float, float]:
        """Grid Search (Full-Fit Mode)."""
        self.logger.info(f"🔍 Tối ưu hóa tham số cho {n_clusters} cụm ({self.center_init.upper()} Mode)")
        
        best_theta = None
        best_alpha = None
        min_distance = float('inf')
        silent_log_level = "ERROR"
        
        for theta in self.theta_values:
            for alpha in self.alpha_values:
                # Truyền center_init vào model con
                model = HedgeAlgebraClustering(
                    n_clusters=n_clusters,
                    theta=theta,
                    alpha=alpha,
                    max_iterations=50, 
                    log_level=silent_log_level,
                    center_init=self.center_init 
                )
                
                result = model.fit(X)
                
                distance = self.calculate_total_distance(
                    result.semantic_values,
                    result.cluster_labels,
                    result.cluster_centers
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_theta = theta
                    best_alpha = alpha
        
        self.logger.info(f"   ✅ Tham số tối ưu: θ={best_theta:.4f}, α={best_alpha:.4f}")
        self.logger.info(f"   📊 Min distance (Squared): {min_distance:.4f}")
        
        return best_theta, best_alpha, min_distance