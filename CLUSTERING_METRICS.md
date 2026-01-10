# Clustering Evaluation Metrics

Module cung cấp các chỉ số đánh giá chất lượng phân cụm dựa trên lý thuyết Fuzzy Clustering.

> **Lưu ý quan trọng (Hard vs Fuzzy):**
> Thuật toán `HedgeAlgebraClustering` gán nhãn cụm theo kiểu **hard** (mỗi điểm thuộc đúng 1 cụm).
> Tuy nhiên, các chỉ số **PC/CE/XB** là chỉ số của **fuzzy clustering** (dựa trên *membership matrix*).
> Vì vậy module sẽ **ước lượng membership matrix** từ khoảng cách đến tâm cụm (tương tự Fuzzy C-Means) để tính metrics
> một cách nhất quán.

---

## Các chỉ số đánh giá

### 1. Partition Coefficient (PC)

**Công thức:**
```
PC = (1/n) * Σᵢ Σⱼ (uᵢⱼ)²
```

**Ý nghĩa:**
- Đo lường mức độ phân biệt và độ tập trung của các cụm mờ
- Giá trị càng cao → cụm càng phân biệt rõ ràng

**Range:** [1/n_clusters, 1]
- **PC = 1**: Các cụm hoàn toàn phân biệt (best)
- **PC = 1/n_clusters**: Các cụm hoàn toàn không phân biệt (worst)

**Ví dụ:**
- PC = 0.98 → Các cụm phân biệt rất tốt
- PC = 0.50 → Các cụm không phân biệt (với 2 cụm)

---

### 2. Classification Entropy (CE)

**Công thức:**
```
CE = -(1/n) * Σᵢ Σⱼ (uᵢⱼ * log(uᵢⱼ))
```

**Ý nghĩa:**
- Đo lường mức độ không chắc chắn trong việc phân cụm
- Giá trị càng thấp → phân cụm càng chắc chắn

**Range:** [0, log(n_clusters)] (thường được normalize về [0, 1])
- **CE = 0**: Phân cụm hoàn toàn chắc chắn (best)
- **CE cao**: Phân cụm không chắc chắn

**Ví dụ:**
- CE = 0.03 → Phân cụm rất chắc chắn
- CE = 0.50 → Phân cụm không chắc chắn

---

### 3. Xie-Beni Index (XB)

**Công thức:**
```
XB = (Σᵢ Σⱼ (uᵢⱼ² * ||xⱼ - vᵢ||²)) / (n * min_{i≠k} ||vᵢ - vₖ||²)
```

**Ý nghĩa:**
- Đo lường chất lượng phân cụm dựa trên tỷ lệ:
  - **Numerator**: Compactness (độ chặt trong từng cụm)
  - **Denominator**: Separation (khoảng cách giữa các cụm)
- Giá trị càng thấp → cụm càng chặt chẽ và tách biệt tốt

**Range:** > 0
- **XB thấp**: Cụm chặt chẽ và tách biệt rõ ràng (best)
- **XB cao**: Cụm lỏng lẻo và chồng chéo (worst)

**Ví dụ:**
- XB = 0.004 → Cụm rất tốt
- XB = 0.5 → Cụm kém chất lượng

---

## Cách sử dụng

### 1. Tự động tính trong ClusteringResult

```python
from src import HedgeAlgebraClustering

clustering = HedgeAlgebraClustering(n_clusters=3)
result = clustering.fit(X)

# Metrics tự động được tính
if result.metrics:
    print(result.metrics.summary())
    print(f"PC: {result.metrics.partition_coefficient:.4f}")
    print(f"CE: {result.metrics.classification_entropy:.4f}")
    print(f"XB: {result.metrics.xie_beni_index:.4f}")
```

### 2. Tính thủ công

```python
from src import ClusteringEvaluator

evaluator = ClusteringEvaluator()
metrics = evaluator.evaluate(X, cluster_labels, cluster_centers)
print(metrics.summary())
```

### 3. Quick evaluate

```python
from src import quick_evaluate

metrics = quick_evaluate(X, cluster_labels, cluster_centers)
print(metrics.summary())
```

### 4. Trong Auto Cluster

```python
from src import AutoClusterPipeline

auto_pipeline = AutoClusterPipeline(min_clusters=2, max_clusters=9)
result = auto_pipeline.run(X)

# Metrics được tính cho mỗi cấu hình
for eval in result.all_evaluations:
    print(
        f"N={eval.n_clusters}: PC={eval.partition_coefficient:.4f}, "
        f"CE={eval.classification_entropy:.4f}, XB={eval.xie_beni_index:.4f}"
    )
```

---

## Gợi ý cho dataset lớn (Sampling)

Tính PC/CE/XB yêu cầu ước lượng membership matrix và có độ phức tạp xấp xỉ **O(n_samples × n_clusters)**.
Với dataset rất lớn, nên evaluate trên một mẫu (sampling) để tiết kiệm thời gian.

```python
import numpy as np
from src import ClusteringEvaluator

evaluator = ClusteringEvaluator()

n = X.shape[0]
sample_size = min(20000, n)
idx = np.random.choice(n, sample_size, replace=False)

metrics = evaluator.evaluate(X[idx], cluster_labels[idx], cluster_centers)
print(metrics.summary())
```

> Silhouette Score trong module thường đã có cơ chế sampling nội bộ.

---

## Ví dụ Output

```
============================================================
📊 CLUSTERING EVALUATION METRICS
============================================================

Partition Coefficient (PC):     0.987999 (↑ cao hơn = tốt hơn)
Classification Entropy (CE):     0.031599 (↓ thấp hơn = tốt hơn)
Xie-Beni Index (XB):            0.003776 (↓ thấp hơn = tốt hơn)
Silhouette Score:                0.930422 (↑ cao hơn = tốt hơn)

============================================================
```

---

## Giải thích kết quả

### Kết quả tốt:
- **PC > 0.8**: Các cụm phân biệt tốt
- **CE < 0.2**: Phân cụm chắc chắn
- **XB < 0.1**: Cụm chặt chẽ và tách biệt

### Kết quả kém:
- **PC < 0.6**: Các cụm không phân biệt
- **CE > 0.5**: Phân cụm không chắc chắn
- **XB > 1.0**: Cụm lỏng lẻo và chồng chéo

---

## Lưu ý

1. **Membership Matrix**: Metrics được tính dựa trên membership matrix (ma trận độ thuộc), ước lượng từ khoảng cách đến tâm cụm theo công thức tương tự Fuzzy C-Means.

2. **Fuzziness Parameter**: Mặc định sử dụng fuzziness = 2.0. Có thể điều chỉnh trong `calculate_membership_matrix()`.

3. **Normalization**: CE có thể được normalize về [0, 1] để dễ so sánh giữa các số cụm khác nhau.

4. **Performance**:
   - Việc tính membership matrix và XB có độ phức tạp ~O(n_samples × n_clusters).
   - Với dataset rất lớn, nên **evaluate trên một mẫu (sampling)**.

5. **Distance space**: Các metrics hiện được tính trong không gian ngữ nghĩa 1D (semantic value), thường là `Sd = mean(X, axis=1)`.

---

## Tài liệu tham khảo

- Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms.
- Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering.

