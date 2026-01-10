"""
Test cases cho Pipeline hoàn chỉnh

Kiểm tra:
- Load dữ liệu CSV
- Tiền xử lý
- Training và Testing
- Đánh giá kết quả
"""

import numpy as np
import pandas as pd
import sys
import tempfile
from pathlib import Path

# Thêm path để import module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import HedgeAlgebraPipeline, quick_run
from src.data_loader import DataLoader
from src.classifier import ClusterClassifier


class TestDataLoader:
    """Test cases cho DataLoader."""
    
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
    
    def create_sample_csv(self, n_samples=100, n_features=10, n_classes=3):
        """Tạo file CSV mẫu để test."""
        np.random.seed(42)
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        columns = [f'feature_{i}' for i in range(n_features)] + ['label']
        data = np.column_stack([X, y])
        df = pd.DataFrame(data, columns=columns)
        
        # Lưu ra file tạm
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        
        return temp_file.name, X, y
    
    def test_load_csv(self):
        """Test load file CSV."""
        print("\n📍 Test: Load CSV")
        
        csv_path, X_expected, y_expected = self.create_sample_csv(50, 5, 2)
        
        loader = DataLoader(log_level="ERROR")
        X, y = loader.load_csv(csv_path, label_column='label')
        
        self.assert_true("Shape đúng", X.shape == (50, 5))
        self.assert_true("Labels đúng", len(y) == 50)
        self.assert_true("Số classes đúng", len(np.unique(y)) == 2)
        
        # Cleanup
        Path(csv_path).unlink()
    
    def test_normalize(self):
        """Test chuẩn hóa dữ liệu."""
        print("\n📍 Test: Chuẩn hóa dữ liệu")
        
        loader = DataLoader(log_level="ERROR")
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_normalized = loader.normalize(X, method="minmax")
        
        self.assert_true("Min = 0", X_normalized.min() >= 0)
        self.assert_true("Max = 1", X_normalized.max() <= 1)
    
    def test_remove_constant_columns(self):
        """Test loại bỏ cột constant."""
        print("\n📍 Test: Loại bỏ cột constant")
        
        loader = DataLoader(log_level="ERROR")
        
        # Cột 1 là constant (tất cả = 5)
        X = np.array([[1, 5, 3], [2, 5, 4], [3, 5, 5]])
        X_filtered = loader.remove_constant_columns(X)
        
        self.assert_true("Số cột giảm", X_filtered.shape[1] == 2)
        self.assert_true("Cột constant bị loại", 1 in loader.removed_columns)
    
    def test_information_gain(self):
        """Test tính Information Gain Ratio."""
        print("\n📍 Test: Tính Information Gain Ratio")
        
        loader = DataLoader(log_level="ERROR")
        
        # Dữ liệu đơn giản
        X = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        y = np.array([0, 0, 1, 1])
        
        ig_ratio = loader.calculate_information_gain_ratio(X, y)
        
        self.assert_true("Có IG ratio cho mỗi feature", len(ig_ratio) == 2)
        self.assert_true("IG ratio >= 0", np.all(ig_ratio >= 0))
    
    def test_split_train_test(self):
        """Test chia train/test."""
        print("\n📍 Test: Chia train/test")
        
        loader = DataLoader(log_level="ERROR")
        
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = loader.split_train_test(
            X, y, test_size=0.2, random_state=42
        )
        
        self.assert_true("Train có 80 samples", len(X_train) == 80)
        self.assert_true("Test có 20 samples", len(X_test) == 20)
    
    def run_all_tests(self):
        """Chạy tất cả test cases."""
        print("=" * 70)
        print("🧪 CHẠY TEST CASES CHO DATA LOADER")
        print("=" * 70)
        
        self.test_load_csv()
        self.test_normalize()
        self.test_remove_constant_columns()
        self.test_information_gain()
        self.test_split_train_test()
        
        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


class TestClassifier:
    """Test cases cho ClusterClassifier."""
    
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
    
    def test_fit_predict(self):
        """Test fit và predict."""
        print("\n📍 Test: Fit và Predict")
        
        np.random.seed(42)
        
        # Tạo dữ liệu phân biệt rõ ràng
        X_train = np.vstack([
            np.random.uniform(0, 0.3, (40, 10)),
            np.random.uniform(0.7, 1.0, (40, 10))
        ])
        y_train = np.array([0] * 40 + [1] * 40)
        
        X_test = np.vstack([
            np.random.uniform(0, 0.3, (10, 10)),
            np.random.uniform(0.7, 1.0, (10, 10))
        ])
        y_test = np.array([0] * 10 + [1] * 10)
        
        classifier = ClusterClassifier(n_clusters=2, log_level="ERROR")
        classifier.fit(X_train, y_train)
        result = classifier.predict(X_test, y_test)
        
        self.assert_true("Predictions có độ dài đúng", len(result.predictions) == 20)
        self.assert_true("Accuracy > 0.5", result.accuracy > 0.5)
        self.assert_true("Có classification report", len(result.classification_report) > 0)
    
    def test_with_multiple_classes(self):
        """Test với nhiều classes."""
        print("\n📍 Test: Nhiều classes")
        
        np.random.seed(42)
        
        X = np.random.rand(150, 5)
        y = np.repeat([0, 1, 2], 50)
        
        classifier = ClusterClassifier(n_clusters=3, log_level="ERROR")
        classifier.fit(X, y)
        
        X_test = np.random.rand(30, 5)
        y_test = np.repeat([0, 1, 2], 10)
        result = classifier.predict(X_test, y_test)
        
        self.assert_true("Có predictions", len(result.predictions) == 30)
    
    def run_all_tests(self):
        """Chạy tất cả test cases."""
        print("=" * 70)
        print("🧪 CHẠY TEST CASES CHO CLASSIFIER")
        print("=" * 70)
        
        self.test_fit_predict()
        self.test_with_multiple_classes()
        
        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


class TestPipeline:
    """Test cases cho HedgeAlgebraPipeline."""
    
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
    
    def create_sample_csv(self, n_samples=100, n_features=10, n_classes=2):
        """Tạo file CSV mẫu."""
        np.random.seed(42)
        
        # Tạo dữ liệu phân biệt theo class
        X_list = []
        y_list = []
        
        for class_id in range(n_classes):
            base = class_id / n_classes
            X_class = np.random.uniform(base, base + 0.3, (n_samples // n_classes, n_features))
            X_list.append(X_class)
            y_list.extend([class_id] * (n_samples // n_classes))
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        columns = [f'feature_{i}' for i in range(n_features)] + ['label']
        data = np.column_stack([X, y])
        df = pd.DataFrame(data, columns=columns)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        
        return temp_file.name
    
    def test_run_with_csv(self):
        """Test chạy pipeline với file CSV."""
        print("\n📍 Test: Chạy pipeline với CSV")
        
        csv_path = self.create_sample_csv(100, 5, 2)
        
        pipeline = HedgeAlgebraPipeline(
            n_clusters=2,
            log_level="WARNING",
            log_to_file=False
        )
        result = pipeline.run(csv_path, label_column='label')
        
        self.assert_true("Có accuracy", result.accuracy >= 0)
        self.assert_true("Có precision", result.precision >= 0)
        self.assert_true("Có recall", result.recall >= 0)
        self.assert_true("Có f1", result.f1 >= 0)
        self.assert_true("Có cluster_centers", len(result.cluster_centers) == 2)
        self.assert_true("Có classification_report", len(result.classification_report) > 0)
        
        # Cleanup
        Path(csv_path).unlink()
    
    def test_run_with_data(self):
        """Test chạy pipeline với dữ liệu có sẵn."""
        print("\n📍 Test: Chạy pipeline với dữ liệu có sẵn")
        
        np.random.seed(42)
        X_train = np.random.rand(80, 5)
        X_test = np.random.rand(20, 5)
        y_train = np.random.randint(0, 2, 80)
        y_test = np.random.randint(0, 2, 20)
        
        pipeline = HedgeAlgebraPipeline(
            n_clusters=2,
            log_level="WARNING",
            log_to_file=False
        )
        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        
        self.assert_true("n_train_samples đúng", result.n_train_samples == 80)
        self.assert_true("n_test_samples đúng", result.n_test_samples == 20)
        self.assert_true("n_features đúng", result.n_features == 5)
        self.assert_true("n_clusters đúng", result.n_clusters == 2)
    
    def test_with_information_gain(self):
        """Test pipeline với Information Gain."""
        print("\n📍 Test: Pipeline với Information Gain")
        
        np.random.seed(42)
        X_train = np.random.rand(80, 5)
        X_test = np.random.rand(20, 5)
        y_train = np.random.randint(0, 2, 80)
        y_test = np.random.randint(0, 2, 20)
        
        pipeline = HedgeAlgebraPipeline(
            n_clusters=2,
            use_information_gain=True,
            log_level="WARNING",
            log_to_file=False
        )
        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        
        self.assert_true("Pipeline hoàn thành", result is not None)
    
    def test_with_optimization(self):
        """Test pipeline với tối ưu hóa tham số."""
        print("\n📍 Test: Pipeline với tối ưu hóa")
        
        np.random.seed(42)
        X_train = np.random.rand(50, 3)
        X_test = np.random.rand(10, 3)
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 10)
        
        pipeline = HedgeAlgebraPipeline(
            n_clusters=2,
            optimize_parameters=True,
            log_level="WARNING",
            log_to_file=False
        )
        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        
        self.assert_true("theta được tối ưu", result.theta > 0)
        self.assert_true("alpha được tối ưu", result.alpha > 0)
    
    def test_different_cluster_counts(self):
        """Test với số cụm khác nhau."""
        print("\n📍 Test: Số cụm khác nhau (2-5)")
        
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        X_test = np.random.rand(20, 5)
        y_train = np.random.randint(0, 3, 100)
        y_test = np.random.randint(0, 3, 20)
        
        for n_clusters in [2, 3, 4, 5]:
            pipeline = HedgeAlgebraPipeline(
                n_clusters=n_clusters,
                log_level="ERROR",
                log_to_file=False
            )
            result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
            
            self.assert_true(
                f"N={n_clusters}: {len(result.cluster_centers)} centers",
                len(result.cluster_centers) == n_clusters
            )
    
    def test_summary(self):
        """Test hàm summary."""
        print("\n📍 Test: Hàm summary")
        
        np.random.seed(42)
        X_train = np.random.rand(50, 3)
        X_test = np.random.rand(10, 3)
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 10)
        
        pipeline = HedgeAlgebraPipeline(
            n_clusters=2,
            log_level="ERROR",
            log_to_file=False
        )
        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        
        summary = result.summary()
        
        self.assert_true("Summary có nội dung", len(summary) > 0)
        self.assert_true("Summary chứa accuracy", "Accuracy" in summary)
    
    def run_all_tests(self):
        """Chạy tất cả test cases."""
        print("=" * 70)
        print("🧪 CHẠY TEST CASES CHO PIPELINE")
        print("=" * 70)
        
        self.test_run_with_csv()
        self.test_run_with_data()
        self.test_with_information_gain()
        self.test_with_optimization()
        self.test_different_cluster_counts()
        self.test_summary()
        
        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


def run_all_tests():
    """Chạy tất cả test cases."""
    print("\n" + "🔬" * 35)
    print("      CHẠY TOÀN BỘ TEST CASES")
    print("🔬" * 35 + "\n")
    
    results = []
    
    # Test DataLoader
    test_data_loader = TestDataLoader()
    results.append(("DataLoader", test_data_loader.run_all_tests()))
    
    # Test Classifier
    test_classifier = TestClassifier()
    results.append(("Classifier", test_classifier.run_all_tests()))
    
    # Test Pipeline
    test_pipeline = TestPipeline()
    results.append(("Pipeline", test_pipeline.run_all_tests()))
    
    # Tổng kết
    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT TOÀN BỘ TESTS")
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

