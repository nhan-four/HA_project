"""
Hedge Algebra Clustering - Module phân cụm dựa trên Đại số gia tử

Module này cung cấp pipeline hoàn chỉnh để:
- Load dữ liệu từ CSV/NPY
- Tiền xử lý và chuẩn hóa dữ liệu
- Phân cụm bằng Đại số gia tử
- Training và testing với các ML models
- Đánh giá và logging kết quả
- Auto cluster từ 2-9 cụm
- Batch processing cho dataset lớn

Tác giả: NguyenNguyenHaoaka.vipro.ahihi
"""

from .config import ClusteringConfig
from .logger import get_logger
from .data_loader import DataLoader
from .clustering import HedgeAlgebraClustering
from .classifier import ClusterClassifier
from .pipeline import HedgeAlgebraPipeline
from .batch_processor import BatchProcessor, MemoryConfig, MemoryManager
from .auto_cluster import AutoClusterPipeline, auto_cluster, AutoClusterResult
from .clustering_metrics import ClusteringEvaluator, ClusteringMetrics, quick_evaluate

__version__ = '2.2.0'
__all__ = [
    'ClusteringConfig',
    'get_logger',
    'DataLoader',
    'HedgeAlgebraClustering',
    'ClusterClassifier',
    'HedgeAlgebraPipeline',
    'BatchProcessor',
    'MemoryConfig',
    'MemoryManager',
    'AutoClusterPipeline',
    'auto_cluster',
    'AutoClusterResult',
    'ClusteringEvaluator',
    'ClusteringMetrics',
    'quick_evaluate'
]

