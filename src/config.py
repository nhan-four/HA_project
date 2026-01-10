"""
Configuration - Cấu hình và constants cho module Hedge Algebra Clustering

File này chứa:
- ClusteringConfig: Class cấu hình chính
- Các constants và default values
"""

from dataclasses import dataclass, field
from typing import Optional, List
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class ClusteringConfig:
    """
    Cấu hình cho toàn bộ pipeline phân cụm.
    
    Attributes:
        n_clusters: Số cụm cần phân (2-10)
        theta: Tham số theta của Đại số gia tử (0 < theta < 1)
        alpha: Tham số alpha của Đại số gia tử (0 < alpha < 1)
        max_iterations: Số vòng lặp tối đa cho clustering
        convergence_tolerance: Ngưỡng hội tụ
        use_information_gain: Có sử dụng Information Gain Ratio không
        random_state: Seed cho reproducibility
        test_size: Tỷ lệ test set (nếu cần split)
        log_level: Mức độ logging (DEBUG, INFO, WARNING, ERROR)
    """
    # Clustering parameters
    n_clusters: int = 2
    theta: float = 0.5
    alpha: float = 0.5
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    
    # Feature selection
    use_information_gain: bool = False
    
    # ML model parameters  
    random_state: int = 42
    
    # Data split
    test_size: float = 0.2
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    
    # Optimization
    optimize_parameters: bool = False
    theta_range: tuple = (0.01, 0.5, 0.01)  # (min, max, step)
    alpha_range: tuple = (0.01, 0.5, 0.01)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Kiểm tra tính hợp lệ của các tham số."""
        if not 2 <= self.n_clusters <= 10:
            raise ValueError(f"n_clusters phải từ 2 đến 10, nhận được: {self.n_clusters}")
        
        if not 0 < self.theta < 1:
            raise ValueError(f"theta phải trong khoảng (0, 1), nhận được: {self.theta}")
        
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha phải trong khoảng (0, 1), nhận được: {self.alpha}")
        
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations phải >= 1, nhận được: {self.max_iterations}")
        
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size phải trong khoảng (0, 1), nhận được: {self.test_size}")
    
    def to_dict(self) -> dict:
        """Chuyển config thành dictionary."""
        return {
            'n_clusters': self.n_clusters,
            'theta': self.theta,
            'alpha': self.alpha,
            'max_iterations': self.max_iterations,
            'convergence_tolerance': self.convergence_tolerance,
            'use_information_gain': self.use_information_gain,
            'random_state': self.random_state,
            'test_size': self.test_size,
            'optimize_parameters': self.optimize_parameters
        }


# Default configuration
DEFAULT_CONFIG = ClusteringConfig()

# Supported number of clusters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10

# Value bounds
MIN_VALUE = 0.0001
MAX_VALUE = 0.9999

