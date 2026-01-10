# Hedge Algebra Clustering

Module phân cụm dựa trên Đại số gia tử (Hedge Algebra) với khả năng:
- Load dữ liệu từ CSV hoặc NPY
- Tiền xử lý tự động (normalize, feature selection)
- Phân cụm bằng Đại số gia tử (mặc định 2–10 cụm, **hỗ trợ N lớn hơn với fallback & warning**)
- **🆕 Dual Init Mode (`center_init`)**: chuyển đổi linh hoạt giữa **Ver6** (mới) và **Legacy** (cũ) để A/B testing
- **🆕 Semantic Scaling**: co giãn tâm cụm lý thuyết `[0,1]` vào dải ngữ nghĩa thực tế `[min(Sd), max(Sd)]` để ổn định hơn trên dữ liệu co cụm
- **🆕 Auto cluster**: Tự động chạy từ 2-9 cụm và chọn cấu hình tốt nhất
- **🆕 Batch processing**: Xử lý dataset lớn (hàng triệu dòng) không tràn RAM
- **🆕 Memory management**: Giới hạn RAM sử dụng
- **🆕 Numpy vectorization**: Tối ưu tốc độ xử lý
- **🆕 Clustering metrics**: Partition Coefficient (PC), Classification Entropy (CE), Xie-Beni Index (XB)
- Training ML model cho từng cụm
- Đánh giá và logging chi tiết

---

## Cài đặt Dependencies

```bash
pip install numpy pandas scikit-learn joblib psutil
```

---

## Cách sử dụng

### 1. Sử dụng cơ bản (với file CSV)

```python
from src import HedgeAlgebraPipeline

pipeline = HedgeAlgebraPipeline(
    n_clusters=3,
    theta=0.5,
    alpha=0.5
)

result = pipeline.run("data.csv", label_column="target")

print(f"Accuracy: {result.accuracy:.4f}")
print(result.summary())
```

### 2. Sử dụng với dữ liệu có sẵn

```python
from src import HedgeAlgebraPipeline

pipeline = HedgeAlgebraPipeline(n_clusters=3)
result = pipeline.run_with_data(X_train, X_test, y_train, y_test)

print(f"Accuracy: {result.accuracy:.4f}")
```

### 3. Sử dụng quick_run (nhanh nhất)

```python
from src.pipeline import quick_run

result = quick_run("data.csv", n_clusters=3, label_column="target")
print(f"Accuracy: {result.accuracy:.4f}")
```

### 4. Tối ưu hóa tham số tự động (Optimizer Full-Fit)

> Optimizer chạy **full-fit** (lặp đến hội tụ) cho mỗi (θ, α) → loss được tính trên trạng thái hội tụ.

```python
from src import HedgeAlgebraPipeline

pipeline = HedgeAlgebraPipeline(
    n_clusters=3,
    optimize_parameters=True
)
result = pipeline.run("data.csv")

print(f"Best theta: {result.theta:.4f}")
print(f"Best alpha: {result.alpha:.4f}")
```

### 5. Sử dụng Information Gain

```python
from src import HedgeAlgebraPipeline

pipeline = HedgeAlgebraPipeline(
    n_clusters=3,
    use_information_gain=True
)
result = pipeline.run("data.csv")
```

### 6. Sử dụng ML model khác

```python
from sklearn.ensemble import RandomForestClassifier
from src import HedgeAlgebraPipeline

pipeline = HedgeAlgebraPipeline(
    n_clusters=3,
    classifier=RandomForestClassifier(n_estimators=100)
)
result = pipeline.run("data.csv")
```

---

## 🆕 Dual Init Mode (`center_init`): Ver6 vs Legacy

Từ Ver 6.5, module hỗ trợ **2 chế độ khởi tạo tâm cụm**:

- `center_init="ver6"` (mặc định): khởi tạo theo **hạng tử ngữ nghĩa** (C/LC/VC + θ) và heuristic đảm bảo thứ tự ngữ nghĩa.
- `center_init="legacy"`: khởi tạo theo **logic code cũ** (tuyến tính a/2, a/4…).

Ví dụ dùng trực tiếp `HedgeAlgebraClustering`:

```python
from src.clustering import HedgeAlgebraClustering

# Mode mới (Ver6) - mặc định
model_ver6 = HedgeAlgebraClustering(n_clusters=6, theta=0.5, alpha=0.5, center_init="ver6")
res1 = model_ver6.fit(X)

# Mode cũ (Legacy)
model_legacy = HedgeAlgebraClustering(n_clusters=6, theta=0.5, alpha=0.5, center_init="legacy")
res2 = model_legacy.fit(X)
```

Tối ưu tham số theo đúng init mode:

```python
from src.clustering import ParameterOptimizer

opt = ParameterOptimizer(center_init="legacy")  # hoặc "ver6"
best_theta, best_alpha, loss = opt.optimize(X, n_clusters=6)
print(best_theta, best_alpha, loss)
```

> Lưu ý: Nếu bạn dùng Pipeline và muốn đổi init mode, hãy kiểm tra Pipeline có forward tham số `center_init` hay không. Nếu chưa, có thể thêm 1 tham số `center_init` vào Pipeline và truyền xuống `HedgeAlgebraClustering`.

---

## 🆕 Auto Cluster (2-9 cụm tự động)

```python
from src import AutoClusterPipeline, auto_cluster

# Cách 1: Hàm tiện ích
result = auto_cluster(X, min_clusters=2, max_clusters=9)
print(f"Best: {result.best_n_clusters} clusters")
print(result.summary())

# Cách 2: Pipeline
auto_pipeline = AutoClusterPipeline(
    min_clusters=2,
    max_clusters=9,
    optimize_params=True
)
result = auto_pipeline.run(X, selection_metric="silhouette")
```

### 🆕 Auto Cluster với Pipeline (Train + Predict)

```python
from src import HedgeAlgebraPipeline

pipeline = HedgeAlgebraPipeline()
result, auto_result = pipeline.run_auto_cluster(
    X_train, X_test, y_train, y_test,
    min_clusters=2,
    max_clusters=9
)

print(f"Best clusters: {auto_result.best_n_clusters}")
print(f"Accuracy: {result.accuracy:.4f}")
```

---

## 🆕 Xử lý Dataset lớn (Batch Processing)

```python
from src import BatchProcessor, LargeDatasetPipeline

# Cách 1: BatchProcessor
processor = BatchProcessor(max_memory_gb=4.0)
for batch_X, batch_y, info in processor.iterate_batches(X, y, batch_size=10000):
    process(batch_X)

# Cách 2: LargeDatasetPipeline
pipeline = LargeDatasetPipeline(max_memory_gb=8.0)
result = pipeline.run("large_data.csv", n_clusters=5, sample_for_training=100000)
```

### 🆕 Quick Auto Run

```python
from src.pipeline import quick_auto_run

result, auto_result = quick_auto_run(
    "data.csv",
    min_clusters=2,
    max_clusters=9,
    label_column="target"
)
```

---

## 🆕 Clustering Evaluation Metrics

```python
from src import ClusteringEvaluator, quick_evaluate
from src.clustering import HedgeAlgebraClustering

# Cách 1: Evaluator
evaluator = ClusteringEvaluator()
metrics = evaluator.evaluate(X, cluster_labels, cluster_centers)
print(metrics.summary())

# Cách 2: Quick evaluate
metrics = quick_evaluate(X, cluster_labels, cluster_centers)

# Metrics tự động trong clustering result
clustering = HedgeAlgebraClustering(n_clusters=3)
result = clustering.fit(X)
if result.metrics:
    print(result.metrics.summary())
```

---

## Cấu trúc Module

```
codebase_project/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── clustering.py
│   ├── classifier.py
│   ├── pipeline.py
│   ├── auto_cluster.py
│   ├── batch_processor.py
│   └── clustering_metrics.py
├── tests/
│   ├── test_clustering.py
│   ├── test_pipeline.py
│   ├── test_auto_cluster.py
│   └── test_clustering_metrics.py
├── examples/
│   └── example_usage.py
└── logs/
```

---

## PipelineResult

| Attribute | Mô tả |
|-----------|-------|
| `accuracy` | Độ chính xác |
| `precision` | Precision (macro) |
| `recall` | Recall (macro) |
| `f1` | F1-score (macro) |
| `training_time` | Thời gian training (s) |
| `testing_time` | Thời gian testing (s) |
| `n_clusters` | Số cụm |
| `cluster_centers` | Tâm các cụm |
| `cluster_distribution` | Phân bố samples trong cụm |
| `classification_report` | Báo cáo phân loại chi tiết |
| `theta` | Tham số theta |
| `alpha` | Tham số alpha |

---

## Chạy Tests

```bash
cd codebase_project

python tests/test_clustering.py
python tests/test_pipeline.py
python tests/test_auto_cluster.py
python tests/test_clustering_metrics.py

python examples/example_usage.py
```

---

## 🆕 AutoClusterResult

| Attribute | Mô tả |
|-----------|-------|
| `best_n_clusters` | Số cụm tốt nhất |
| `best_evaluation` | Đánh giá của cấu hình tốt nhất |
| `all_evaluations` | Danh sách đánh giá tất cả cấu hình |
| `total_time` | Tổng thời gian chạy |

---

## 🆕 Memory Configuration

```python
from src import BatchProcessor, MemoryConfig

config = MemoryConfig(
    max_memory_gb=4.0,
    batch_size=10000,
    reserve_memory_gb=1.0
)

processor = BatchProcessor(max_memory_gb=4.0)
```

---

## 🆕 Clustering Evaluation Metrics

Module cung cấp 3 chỉ số đánh giá phân cụm chính:

### 1. Partition Coefficient (PC)
- **Ý nghĩa**: mức độ phân biệt và độ tập trung của các cụm mờ
- **Range**: [1/n_clusters, 1]
- **Càng cao càng tốt**

### 2. Classification Entropy (CE)
- **Ý nghĩa**: mức độ không chắc chắn trong phân cụm
- **Range**: [0, 1] (normalized)
- **Càng thấp càng tốt**

### 3. Xie-Beni Index (XB)
- **Ý nghĩa**: tỷ lệ compactness/separation
- **Range**: > 0
- **Càng thấp càng tốt**

Ví dụ output:

```
============================================================
📊 CLUSTERING EVALUATION METRICS
============================================================

Partition Coefficient (PC):     0.823456 (↑ cao hơn = tốt hơn)
Classification Entropy (CE):     0.234567 (↓ thấp hơn = tốt hơn)
Xie-Beni Index (XB):             0.012345 (↓ thấp hơn = tốt hơn)
Silhouette Score:                0.789012 (↑ cao hơn = tốt hơn)

============================================================
```

---

## Lý thuyết Đại số gia tử (tóm tắt)

Đại số gia tử (Hedge Algebra) là framework toán học để biểu diễn các khái niệm mờ như "rất cao", "khá thấp", "trung bình".

### Luồng phân cụm (tóm tắt)
1) Chuẩn hóa dữ liệu → tính giá trị ngữ nghĩa `Sd_i` (mean theo sample)
2) Khởi tạo tâm cụm `SC_k` (Ver6 hoặc Legacy)
3) Gán cụm theo midpoint: nếu `x <= midpoint` → cụm trái
4) Cập nhật tâm = trung bình `Sd_i` trong cụm (cụm rỗng giữ tâm cũ)
5) Lặp đến hội tụ

### Semantic Scaling
Tâm cụm lý thuyết `[0,1]` được map vào dải thực tế `[min(Sd), max(Sd)]` để giảm rủi ro cụm rỗng khi dữ liệu co cụm.

---

## Tác giả

Nguyen Van Nhan
ICN-Lab

## License

MIT

