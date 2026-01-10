"""
Test cases cho Auto Cluster và Batch Processing

Kiểm tra:
- AutoClusterPipeline
- BatchProcessor
- MemoryManager
- Tối ưu hóa vectorization
"""

import numpy as np
import sys
from pathlib import Path

# Thêm path để import module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_cluster import (
    AutoClusterPipeline,
    auto_cluster,
    OptimizedClustering,
    AutoClusterResult
)
from src.batch_processor import (
    BatchProcessor,
    MemoryManager,
    MemoryConfig,
    estimate_memory_requirement
)


class TestOptimizedClustering:
    """Test cho OptimizedClustering (vectorized operations)."""
    
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
    
    def assert_close(self, name: str, a, b, tol=1e-6):
        if isinstance(a, np.ndarray):
            condition = np.allclose(a, b, atol=tol)
        else:
            condition = abs(a - b) < tol
        self.assert_true(name, condition)
    
    def test_semantic_values_vectorized(self):
        """Test tính semantic values với vectorization."""
        print("\n📍 Test: Semantic values vectorized")
        
        X = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        result = OptimizedClustering.calculate_semantic_values_vectorized(X)
        expected = np.array([0.2, 0.5, 0.8])
        
        self.assert_close("Semantic values đúng", result, expected)
    
    def test_assign_clusters_vectorized(self):
        """Test gán cụm với vectorization."""
        print("\n📍 Test: Assign clusters vectorized")
        
        semantic_values = np.array([0.1, 0.3, 0.6, 0.9])
        centers = np.array([0.25, 0.75])
        
        labels = OptimizedClustering.assign_to_clusters_vectorized(semantic_values, centers)
        expected = np.array([1, 1, 2, 2])
        
        self.assert_close("Labels đúng (2 cụm)", labels, expected)
        
        # Test với 3 cụm
        centers = np.array([0.2, 0.5, 0.8])
        semantic_values = np.array([0.1, 0.3, 0.5, 0.55, 0.7, 0.9])
        labels = OptimizedClustering.assign_to_clusters_vectorized(semantic_values, centers)
        
        self.assert_true("Labels có 3 giá trị khác nhau", len(np.unique(labels)) == 3)
    
    def test_update_centers_vectorized(self):
        """Test cập nhật centers với vectorization."""
        print("\n📍 Test: Update centers vectorized")
        
        semantic_values = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 2, 2, 2])
        
        new_centers = OptimizedClustering.update_centers_vectorized(semantic_values, labels, 2, np.array([0.0, 0.0]))
        expected = np.array([0.2, 0.8])
        
        self.assert_close("Centers được cập nhật đúng", new_centers, expected)
    
    def test_silhouette_fast(self):
        """Test tính Silhouette score nhanh."""
        print("\n📍 Test: Silhouette score fast")
        
        # Dữ liệu phân biệt rõ ràng
        np.random.seed(42)
        semantic_values = np.concatenate([
            np.random.uniform(0, 0.3, 50),
            np.random.uniform(0.7, 1.0, 50)
        ])
        labels = np.array([1] * 50 + [2] * 50)
        centers = np.array([0.15, 0.85])
        
        score = OptimizedClustering.calculate_silhouette_fast(
            semantic_values, labels, centers
        )
        
        self.assert_true("Silhouette > 0 cho dữ liệu phân biệt", score > 0)
        self.assert_true("Silhouette <= 1", score <= 1)
    
    def run_all_tests(self):
        """Chạy tất cả tests."""
        print("=" * 70)
        print("🧪 TEST OPTIMIZED CLUSTERING (VECTORIZED)")
        print("=" * 70)
        
        self.test_semantic_values_vectorized()
        self.test_assign_clusters_vectorized()
        self.test_update_centers_vectorized()
        self.test_silhouette_fast()
        
        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


class TestBatchProcessor:
    """Test cho BatchProcessor."""
    
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
    
    def test_iterate_batches(self):
        """Test iterate qua batches."""
        print("\n📍 Test: Iterate batches")
        
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        processor = BatchProcessor(max_memory_gb=1.0, log_level="ERROR")
        
        batches = list(processor.iterate_batches(X, y, batch_size=30))
        
        self.assert_true("Có 4 batches (100/30)", len(batches) == 4)
        
        # Kiểm tra tổng samples
        total_samples = sum(b[2]['n_samples'] for b in batches)
        self.assert_true("Tổng samples = 100", total_samples == 100)
    
    def test_process_in_batches(self):
        """Test xử lý theo batch."""
        print("\n📍 Test: Process in batches")
        
        X = np.random.rand(100, 5)
        processor = BatchProcessor(max_memory_gb=1.0, log_level="ERROR")
        
        # Tính mean theo batch
        result = processor.process_in_batches(
            X,
            lambda batch: np.mean(batch, axis=1),
            batch_size=30
        )
        
        # So sánh với kết quả không batch
        expected = np.mean(X, axis=1)
        
        self.assert_true("Kết quả giống khi không batch", np.allclose(result, expected))
    
    def test_semantic_values_batched(self):
        """Test tính semantic values theo batch."""
        print("\n📍 Test: Semantic values batched")
        
        X = np.random.rand(1000, 10)
        processor = BatchProcessor(max_memory_gb=1.0, log_level="ERROR")
        
        result_batched = processor.calculate_semantic_values_batched(X, batch_size=200)
        result_normal = np.mean(X, axis=1)
        
        self.assert_true("Kết quả batch == kết quả thường", np.allclose(result_batched, result_normal))
    
    def test_memory_estimation(self):
        """Test ước tính memory."""
        print("\n📍 Test: Memory estimation")
        
        mem_gb = estimate_memory_requirement(
            n_samples=1000000,
            n_features=100,
            n_clusters=5
        )
        
        self.assert_true("Memory > 0", mem_gb > 0)
        self.assert_true("Memory hợp lý (<100GB cho 1M samples)", mem_gb < 100)
    
    def run_all_tests(self):
        """Chạy tất cả tests."""
        print("=" * 70)
        print("🧪 TEST BATCH PROCESSOR")
        print("=" * 70)
        
        self.test_iterate_batches()
        self.test_process_in_batches()
        self.test_semantic_values_batched()
        self.test_memory_estimation()
        
        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


class TestAutoCluster:
    """Test cho AutoClusterPipeline."""
    
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
    
    def test_auto_cluster_basic(self):
        """Test auto cluster cơ bản."""
        print("\n📍 Test: Auto cluster cơ bản")
        
        np.random.seed(42)
        # Tạo dữ liệu với 3 clusters rõ ràng
        X = np.vstack([
            np.random.uniform(0, 0.2, (50, 5)),
            np.random.uniform(0.4, 0.6, (50, 5)),
            np.random.uniform(0.8, 1.0, (50, 5))
        ])
        
        result = auto_cluster(X, min_clusters=2, max_clusters=5)
        
        self.assert_true("Có kết quả", result is not None)
        self.assert_true("best_n_clusters trong range", 2 <= result.best_n_clusters <= 5)
        self.assert_true("Có evaluations cho tất cả cụm", len(result.all_evaluations) == 4)
    
    def test_auto_cluster_selection_metrics(self):
        """Test các metrics khác nhau để chọn cụm."""
        print("\n📍 Test: Selection metrics")
        
        np.random.seed(42)
        X = np.random.rand(100, 5)
        
        pipeline = AutoClusterPipeline(
            min_clusters=2,
            max_clusters=4,
            optimize_params=False,  # Tắt để nhanh
            log_level="ERROR"
        )
        
        # Test với silhouette
        result_sil = pipeline.run(X, selection_metric="silhouette")
        self.assert_true("Silhouette metric hoạt động", result_sil is not None)
        
        # Test với distance
        result_dist = pipeline.run(X, selection_metric="distance")
        self.assert_true("Distance metric hoạt động", result_dist is not None)
    
    def test_auto_cluster_summary(self):
        """Test hàm summary."""
        print("\n📍 Test: Summary")
        
        np.random.seed(42)
        X = np.random.rand(50, 3)
        
        result = auto_cluster(X, min_clusters=2, max_clusters=3)
        
        summary = result.summary()
        self.assert_true("Summary có nội dung", len(summary) > 0)
        self.assert_true("Summary chứa 'KẾT QUẢ'", "KẾT QUẢ" in summary)
    
    def run_all_tests(self):
        """Chạy tất cả tests."""
        print("=" * 70)
        print("🧪 TEST AUTO CLUSTER")
        print("=" * 70)
        
        self.test_auto_cluster_basic()
        self.test_auto_cluster_selection_metrics()
        self.test_auto_cluster_summary()
        
        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


def run_all_tests():
    """Chạy tất cả test cases."""
    print("\n" + "🔬" * 35)
    print("      CHẠY TEST AUTO CLUSTER & BATCH PROCESSING")
    print("🔬" * 35 + "\n")
    
    results = []
    
    # Test OptimizedClustering
    test_opt = TestOptimizedClustering()
    results.append(("OptimizedClustering", test_opt.run_all_tests()))
    
    # Test BatchProcessor
    test_batch = TestBatchProcessor()
    results.append(("BatchProcessor", test_batch.run_all_tests()))
    
    # Test AutoCluster
    test_auto = TestAutoCluster()
    results.append(("AutoCluster", test_auto.run_all_tests()))
    
    # Tổng kết
    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT TESTS")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 TẤT CẢ TEST CASES ĐỀU PASS!")
    else:
        print("\n⚠️ CÓ TEST CASES THẤT BẠI!")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()

