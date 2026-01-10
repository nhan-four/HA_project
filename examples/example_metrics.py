"""
Ví dụ sử dụng Clustering Evaluation Metrics

Minh họa cách sử dụng các chỉ số đánh giá phân cụm:
- Partition Coefficient (PC)
- Classification Entropy (CE)
- Xie-Beni Index (XB)
"""

import numpy as np
import sys
from pathlib import Path

# Thêm path để import module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    HedgeAlgebraClustering,
    ClusteringEvaluator,
    quick_evaluate,
    AutoClusterPipeline
)


def example_1_basic_metrics():
    """
    Ví dụ 1: Tính metrics cơ bản
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 1: TÍNH METRICS CƠ BẢN")
    print("=" * 70)
    
    # Tạo dữ liệu với 3 cụm rõ ràng
    np.random.seed(42)
    X = np.vstack([
        np.random.uniform(0, 0.2, (30, 5)),
        np.random.uniform(0.4, 0.6, (30, 5)),
        np.random.uniform(0.8, 1.0, (30, 5))
    ])
    
    # Phân cụm
    clustering = HedgeAlgebraClustering(n_clusters=3, log_level="ERROR")
    result = clustering.fit(X)
    
    # Metrics tự động được tính trong result
    if result.metrics:
        print("\n📊 Metrics tự động từ clustering result:")
        print(result.metrics.summary())
    
    # Hoặc tính thủ công
    evaluator = ClusteringEvaluator(log_level="ERROR")
    metrics = evaluator.evaluate(X, result.cluster_labels, result.cluster_centers)
    
    print("\n📊 Metrics tính thủ công:")
    print(metrics.summary())


def example_2_compare_clusters():
    """
    Ví dụ 2: So sánh metrics với số cụm khác nhau
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 2: SO SÁNH METRICS VỚI SỐ CỤM KHÁC NHAU")
    print("=" * 70)
    
    np.random.seed(42)
    X = np.random.rand(100, 5)
    
    evaluator = ClusteringEvaluator(log_level="ERROR")
    
    print("\n📊 So sánh metrics:")
    print("-" * 80)
    print(f"{'N':>3} | {'PC':>10} | {'CE':>10} | {'XB':>12} | {'Silhouette':>12}")
    print("-" * 80)
    
    for n_clusters in [2, 3, 4, 5]:
        clustering = HedgeAlgebraClustering(n_clusters=n_clusters, log_level="ERROR")
        result = clustering.fit(X)
        
        if result.metrics:
            metrics = result.metrics
            print(
                f"{n_clusters:>3} | "
                f"{metrics.partition_coefficient:>10.4f} | "
                f"{metrics.classification_entropy:>10.4f} | "
                f"{metrics.xie_beni_index:>12.4f} | "
                f"{metrics.silhouette_score:>12.4f}"
            )
    
    print("-" * 80)


def example_3_auto_cluster_with_metrics():
    """
    Ví dụ 3: Auto cluster với metrics
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 3: AUTO CLUSTER VỚI METRICS")
    print("=" * 70)
    
    np.random.seed(42)
    X = np.vstack([
        np.random.uniform(0, 0.2, (40, 5)),
        np.random.uniform(0.4, 0.6, (40, 5)),
        np.random.uniform(0.8, 1.0, (40, 5))
    ])
    
    auto_pipeline = AutoClusterPipeline(
        min_clusters=2,
        max_clusters=5,
        optimize_params=False,  # Tắt để nhanh
        log_level="INFO"
    )
    
    result = auto_pipeline.run(X, selection_metric="silhouette")
    
    print("\n📊 Metrics của cụm tốt nhất:")
    best_eval = result.best_evaluation
    print(f"   PC: {best_eval.partition_coefficient:.4f}")
    print(f"   CE: {best_eval.classification_entropy:.4f}")
    print(f"   XB: {best_eval.xie_beni_index:.4f}")
    print(f"   Silhouette: {best_eval.silhouette_score:.4f}")


def example_4_quick_evaluate():
    """
    Ví dụ 4: Sử dụng quick_evaluate
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 4: QUICK EVALUATE")
    print("=" * 70)
    
    np.random.seed(42)
    X = np.random.rand(50, 5)
    
    # Phân cụm
    clustering = HedgeAlgebraClustering(n_clusters=3, log_level="ERROR")
    result = clustering.fit(X)
    
    # Quick evaluate
    metrics = quick_evaluate(X, result.cluster_labels, result.cluster_centers)
    
    print("\n📊 Kết quả quick_evaluate:")
    print(metrics.summary())


def example_5_interpretation():
    """
    Ví dụ 5: Giải thích ý nghĩa các metrics
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 5: GIẢI THÍCH Ý NGHĨA METRICS")
    print("=" * 70)
    
    print("""
📊 PARTITION COEFFICIENT (PC):
   • Range: [1/n_clusters, 1]
   • PC cao → Các cụm phân biệt rõ ràng
   • PC = 1 → Hoàn toàn phân biệt (best)
   • PC = 1/n_clusters → Không phân biệt (worst)

📊 CLASSIFICATION ENTROPY (CE):
   • Range: [0, 1] (normalized)
   • CE thấp → Phân cụm chắc chắn
   • CE = 0 → Hoàn toàn chắc chắn (best)
   • CE = 1 → Hoàn toàn không chắc chắn (worst)

📊 XIE-BENI INDEX (XB):
   • Range: > 0
   • XB thấp → Cụm chặt chẽ và tách biệt tốt
   • XB = compactness / separation
   • Càng thấp càng tốt

📊 SILHOUETTE SCORE:
   • Range: [-1, 1]
   • Score cao → Cụm phân biệt tốt
   • Score = 1 → Hoàn hảo (best)
   • Score = -1 → Tệ nhất (worst)
    """)


if __name__ == "__main__":
    print("\n" + "📊" * 35)
    print("      CLUSTERING EVALUATION METRICS - VÍ DỤ SỬ DỤNG")
    print("📊" * 35)
    
    example_1_basic_metrics()
    example_2_compare_clusters()
    example_3_auto_cluster_with_metrics()
    example_4_quick_evaluate()
    example_5_interpretation()
    
    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT TẤT CẢ CÁC VÍ DỤ")
    print("=" * 70)

