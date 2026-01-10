"""
Ví dụ sử dụng Hedge Algebra Clustering Pipeline

File này minh họa các cách sử dụng module:
1. Sử dụng cơ bản với file CSV
2. Sử dụng với dữ liệu có sẵn
3. Sử dụng với tối ưu hóa tham số
4. Sử dụng với Information Gain
5. Sử dụng với các ML models khác nhau
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Thêm path để import module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import HedgeAlgebraPipeline
from src.pipeline import quick_run


def create_sample_data():
    """Tạo dữ liệu mẫu để demo."""
    np.random.seed(42)
    
    n_samples = 200
    n_features = 10
    n_classes = 3
    
    # Tạo dữ liệu phân biệt theo class
    X_list = []
    y_list = []
    
    samples_per_class = n_samples // n_classes
    for class_id in range(n_classes):
        base = class_id * 0.3
        X_class = np.random.uniform(base, base + 0.25, (samples_per_class, n_features))
        X_list.append(X_class)
        y_list.extend([class_id] * samples_per_class)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    return X, y


def example_1_basic_usage():
    """
    Ví dụ 1: Sử dụng cơ bản
    
    Đây là cách đơn giản nhất để sử dụng pipeline.
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 1: SỬ DỤNG CƠ BẢN")
    print("=" * 70)
    
    # Tạo dữ liệu
    X, y = create_sample_data()
    
    # Chia train/test
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Dữ liệu:")
    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   Classes: {np.unique(y)}")
    
    # Khởi tạo và chạy pipeline
    pipeline = HedgeAlgebraPipeline(
        n_clusters=3,           # Số cụm
        theta=0.5,              # Tham số theta
        alpha=0.5,              # Tham số alpha
        log_level="INFO",       # Mức độ logging
        log_to_file=False       # Không ghi log ra file
    )
    
    # Chạy với dữ liệu có sẵn
    result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
    
    # In kết quả
    print(result.summary())


def example_2_with_csv():
    """
    Ví dụ 2: Sử dụng với file CSV
    
    Pipeline tự động load, tiền xử lý và train.
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 2: SỬ DỤNG VỚI FILE CSV")
    print("=" * 70)
    
    # Tạo file CSV mẫu
    X, y = create_sample_data()
    columns = [f'feature_{i}' for i in range(X.shape[1])] + ['target']
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=columns)
    
    csv_path = "/tmp/sample_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📁 Đã tạo file CSV: {csv_path}")
    print(f"   Shape: {df.shape}")
    
    # Chạy pipeline với file CSV
    pipeline = HedgeAlgebraPipeline(
        n_clusters=3,
        log_level="INFO",
        log_to_file=False
    )
    
    result = pipeline.run(
        file_path=csv_path,
        label_column='target',      # Tên cột label
        normalize_method='minmax'   # Phương pháp chuẩn hóa
    )
    
    print(result.summary())
    
    # Cleanup
    Path(csv_path).unlink()


def example_3_with_optimization():
    """
    Ví dụ 3: Sử dụng với tối ưu hóa tham số
    
    Pipeline tự động tìm theta và alpha tối ưu.
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 3: TỐI ƯU HÓA THAM SỐ")
    print("=" * 70)
    
    X, y = create_sample_data()
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    pipeline = HedgeAlgebraPipeline(
        n_clusters=3,
        optimize_parameters=True,   # Bật tối ưu hóa
        log_level="INFO",
        log_to_file=False
    )
    
    result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
    
    print(f"\n📊 Tham số được tối ưu:")
    print(f"   Theta: {result.theta:.4f}")
    print(f"   Alpha: {result.alpha:.4f}")
    print(result.summary())


def example_4_with_information_gain():
    """
    Ví dụ 4: Sử dụng với Information Gain
    
    Sử dụng IG Ratio làm trọng số cho features.
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 4: SỬ DỤNG INFORMATION GAIN")
    print("=" * 70)
    
    X, y = create_sample_data()
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Tính IG ratio từ DataLoader
    from src.data_loader import DataLoader
    
    loader = DataLoader(log_level="INFO")
    ig_weights = loader.calculate_information_gain_ratio(X_train, y_train)
    print(f"\n📊 Information Gain Ratio: {ig_weights}")
    
    pipeline = HedgeAlgebraPipeline(
        n_clusters=3,
        use_information_gain=True,
        log_level="INFO",
        log_to_file=False
    )
    
    result = pipeline.run_with_data(
        X_train, X_test, y_train, y_test,
        information_gain_weights=ig_weights
    )
    
    print(result.summary())


def example_5_different_classifiers():
    """
    Ví dụ 5: Sử dụng với các ML models khác nhau
    
    Có thể dùng bất kỳ sklearn classifier nào.
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 5: CÁC ML MODELS KHÁC NHAU")
    print("=" * 70)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    
    X, y = create_sample_data()
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n🔧 Testing: {name}")
        
        pipeline = HedgeAlgebraPipeline(
            n_clusters=3,
            classifier=clf,
            log_level="WARNING",
            log_to_file=False
        )
        
        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        results[name] = result.accuracy
        
        print(f"   Accuracy: {result.accuracy:.4f}")
    
    # So sánh kết quả
    print("\n" + "=" * 50)
    print("📊 SO SÁNH KẾT QUẢ")
    print("=" * 50)
    
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {name}: {acc:.4f}")


def example_6_quick_run():
    """
    Ví dụ 6: Sử dụng quick_run
    
    Cách nhanh nhất để chạy pipeline.
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 6: QUICK RUN")
    print("=" * 70)
    
    # Tạo file CSV
    X, y = create_sample_data()
    columns = [f'f{i}' for i in range(X.shape[1])] + ['label']
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=columns)
    
    csv_path = "/tmp/quick_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Quick run
    result = quick_run(csv_path, n_clusters=3, label_column='label')
    
    print(result.summary())
    
    # Cleanup
    Path(csv_path).unlink()


def example_7_different_cluster_counts():
    """
    Ví dụ 7: So sánh số cụm khác nhau
    """
    print("\n" + "=" * 70)
    print("📌 VÍ DỤ 7: SO SÁNH SỐ CỤM KHÁC NHAU")
    print("=" * 70)
    
    X, y = create_sample_data()
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    results = {}
    
    for n_clusters in [2, 3, 4, 5]:
        pipeline = HedgeAlgebraPipeline(
            n_clusters=n_clusters,
            log_level="ERROR",
            log_to_file=False
        )
        
        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        results[n_clusters] = {
            'accuracy': result.accuracy,
            'f1': result.f1,
            'centers': result.cluster_centers
        }
    
    print("\n📊 SO SÁNH ACCURACY VỚI SỐ CỤM KHÁC NHAU")
    print("-" * 50)
    
    for n, r in results.items():
        centers_str = ", ".join([f"{c:.3f}" for c in r['centers']])
        print(f"   {n} cụm: Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}")
        print(f"          Centers: [{centers_str}]")


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("      HEDGE ALGEBRA CLUSTERING - VÍ DỤ SỬ DỤNG")
    print("🚀" * 35)
    
    # Chạy các ví dụ
    example_1_basic_usage()
    example_2_with_csv()
    example_3_with_optimization()
    example_4_with_information_gain()
    example_5_different_classifiers()
    example_6_quick_run()
    example_7_different_cluster_counts()
    
    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT TẤT CẢ CÁC VÍ DỤ")
    print("=" * 70)

