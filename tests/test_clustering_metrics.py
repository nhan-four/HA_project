"""
Test cases cho Clustering Metrics

Kiểm tra:
- Partition Coefficient (PC)
- Classification Entropy (CE)
- Xie-Beni Index (XB)
- Silhouette Score
"""

import numpy as np
import sys
from pathlib import Path

# Thêm path để import module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering_metrics import (
    ClusteringEvaluator,
    ClusteringMetrics,
    quick_evaluate
)


class TestClusteringMetrics:
    """Test cases cho Clustering Metrics."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_true(self, name: str, condition: bool):
        if condition:
            self.passed += 1
            print(f"   ✅ {name}")
        else:
            self.failed += 1
            print(f"   ❌ {name}")
    
    def assert_in_range(self, name: str, value: float, min_val: float, max_val: float):
        condition = min_val <= value <= max_val
        self.assert_true(f"{name} trong range [{min_val}, {max_val}]", condition)
    
    def test_partition_coefficient(self):
        """Test Partition Coefficient (PC)."""
        print("\n📍 Test: Partition Coefficient (PC)")
        
        evaluator = ClusteringEvaluator(log_level="ERROR")
        
        # Test với membership matrix đơn giản
        # 2 cụm, 4 samples
        membership = np.array([
            [0.9, 0.8, 0.2, 0.1],  # Cụm 1
            [0.1, 0.2, 0.8, 0.9]   # Cụm 2
        ])
        
        pc = evaluator.calculate_partition_coefficient(membership)
        
        # PC phải trong [1/n_clusters, 1] = [0.5, 1]
        self.assert_in_range("PC", pc, 0.5, 1.0)
        
        # Test với membership hoàn toàn phân biệt
        membership_perfect = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
        pc_perfect = evaluator.calculate_partition_coefficient(membership_perfect)
        self.assert_true("PC = 1.0 khi hoàn toàn phân biệt", abs(pc_perfect - 1.0) < 1e-6)
        
        # Test với membership không phân biệt
        membership_uniform = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5]
        ])
        pc_uniform = evaluator.calculate_partition_coefficient(membership_uniform)
        self.assert_true("PC = 0.5 khi không phân biệt", abs(pc_uniform - 0.5) < 1e-6)
    
    def test_classification_entropy(self):
        """Test Classification Entropy (CE)."""
        print("\n📍 Test: Classification Entropy (CE)")
        
        evaluator = ClusteringEvaluator(log_level="ERROR")
        
        # Test với membership đơn giản
        membership = np.array([
            [0.9, 0.8, 0.2, 0.1],
            [0.1, 0.2, 0.8, 0.9]
        ])
        
        ce = evaluator.calculate_classification_entropy(membership)
        
        # CE normalized phải trong [0, 1]
        self.assert_in_range("CE", ce, 0.0, 1.0)
        
        # Test với membership hoàn toàn chắc chắn
        membership_certain = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
        ce_certain = evaluator.calculate_classification_entropy(membership_certain)
        self.assert_true("CE = 0 khi hoàn toàn chắc chắn", ce_certain < 0.01)
    
    def test_xie_beni_index(self):
        """Test Xie-Beni Index (XB)."""
        print("\n📍 Test: Xie-Beni Index (XB)")
        
        evaluator = ClusteringEvaluator(log_level="ERROR")
        
        # Tạo dữ liệu phân biệt rõ ràng
        np.random.seed(42)
        X_low = np.random.uniform(0, 0.2, (20, 5))
        X_high = np.random.uniform(0.8, 1.0, (20, 5))
        X = np.vstack([X_low, X_high])
        
        semantic_values = np.mean(X, axis=1)
        centers = np.array([0.1, 0.9])
        labels = np.array([1] * 20 + [2] * 20)
        
        membership = evaluator.calculate_membership_matrix(semantic_values, centers)
        xb = evaluator.calculate_xie_beni_index(X, membership, centers)
        
        # XB phải > 0
        self.assert_true("XB > 0", xb > 0)
        
        # XB thấp cho dữ liệu phân biệt tốt
        self.assert_true("XB hợp lý cho dữ liệu phân biệt", xb < 100)
    
    def test_membership_matrix(self):
        """Test tính membership matrix."""
        print("\n📍 Test: Membership Matrix")
        
        evaluator = ClusteringEvaluator(log_level="ERROR")
        
        semantic_values = np.array([0.1, 0.3, 0.7, 0.9])
        centers = np.array([0.2, 0.8])
        
        membership = evaluator.calculate_membership_matrix(semantic_values, centers)
        
        # Kiểm tra shape
        self.assert_true("Shape đúng (n_clusters, n_samples)", membership.shape == (2, 4))
        
        # Kiểm tra tổng mỗi cột = 1
        column_sums = np.sum(membership, axis=0)
        self.assert_true("Tổng mỗi cột = 1", np.allclose(column_sums, 1.0, atol=1e-6))
        
        # Kiểm tra giá trị trong [0, 1]
        self.assert_true("Values trong [0, 1]", np.all((membership >= 0) & (membership <= 1)))
    
    def test_full_evaluation(self):
        """Test đánh giá đầy đủ."""
        print("\n📍 Test: Full Evaluation")
        
        evaluator = ClusteringEvaluator(log_level="ERROR")
        
        # Tạo dữ liệu với 3 cụm rõ ràng
        np.random.seed(42)
        X = np.vstack([
            np.random.uniform(0, 0.2, (30, 5)),
            np.random.uniform(0.4, 0.6, (30, 5)),
            np.random.uniform(0.8, 1.0, (30, 5))
        ])
        
        labels = np.array([1] * 30 + [2] * 30 + [3] * 30)
        centers = np.array([0.1, 0.5, 0.9])
        
        metrics = evaluator.evaluate(X, labels, centers)
        
        # Kiểm tra tất cả metrics
        self.assert_in_range("PC", metrics.partition_coefficient, 0.0, 1.0)
        self.assert_in_range("CE", metrics.classification_entropy, 0.0, 1.0)
        self.assert_true("XB > 0", metrics.xie_beni_index > 0)
        self.assert_in_range("Silhouette", metrics.silhouette_score, -1.0, 1.0)
        
        # Kiểm tra summary
        summary = metrics.summary()
        self.assert_true("Summary có nội dung", len(summary) > 0)
        self.assert_true("Summary chứa PC", "Partition Coefficient" in summary)
        self.assert_true("Summary chứa CE", "Classification Entropy" in summary)
        self.assert_true("Summary chứa XB", "Xie-Beni Index" in summary)
    
    def test_quick_evaluate(self):
        """Test hàm quick_evaluate."""
        print("\n📍 Test: Quick Evaluate")
        
        np.random.seed(42)
        X = np.random.rand(50, 5)
        labels = np.array([1] * 25 + [2] * 25)
        centers = np.array([0.3, 0.7])
        
        metrics = quick_evaluate(X, labels, centers)
        
        self.assert_true("Có metrics", metrics is not None)
        self.assert_in_range("PC", metrics.partition_coefficient, 0.0, 1.0)
    
    def test_edge_cases(self):
        """Test edge cases."""
        print("\n📍 Test: Edge Cases")
        
        evaluator = ClusteringEvaluator(log_level="ERROR")
        
        # Test với 1 cụm
        X = np.random.rand(10, 3)
        labels = np.ones(10)
        centers = np.array([0.5])
        
        try:
            metrics = evaluator.evaluate(X, labels, centers, calculate_silhouette=False)
            self.assert_true("Xử lý được 1 cụm", True)
        except Exception:
            self.assert_true("Xử lý được 1 cụm", False)
        
        # Test với dữ liệu giống nhau
        X_identical = np.ones((10, 3)) * 0.5
        labels = np.array([1] * 5 + [2] * 5)
        centers = np.array([0.5, 0.5])
        
        try:
            metrics = evaluator.evaluate(X_identical, labels, centers, calculate_silhouette=False)
            self.assert_true("Xử lý được dữ liệu giống nhau", True)
        except Exception:
            self.assert_true("Xử lý được dữ liệu giống nhau", False)
    
    def run_all_tests(self):
        """Chạy tất cả tests."""
        print("=" * 70)
        print("🧪 TEST CLUSTERING METRICS")
        print("=" * 70)
        
        self.test_partition_coefficient()
        self.test_classification_entropy()
        self.test_xie_beni_index()
        self.test_membership_matrix()
        self.test_full_evaluation()
        self.test_quick_evaluate()
        self.test_edge_cases()
        
        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


if __name__ == "__main__":
    test = TestClusteringMetrics()
    success = test.run_all_tests()
    
    if success:
        print("\n🎉 TẤT CẢ TEST CASES ĐỀU PASS!")
    else:
        print("\n⚠️ CÓ TEST CASES THẤT BẠI!")

