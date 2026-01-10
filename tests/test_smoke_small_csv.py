import os
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.pipeline import HedgeAlgebraPipeline
from src.auto_cluster import AutoClusterPipeline


class TestSmokeSmallCSV(unittest.TestCase):
    CSV_PATH = "/home/nhannv/Hello/ICN/data_process/dataset_5percent_33classes_no_normalize.csv"
    LABEL_COL = "label"
    SAMPLE_ROWS = 20000
    MIN_PER_CLASS = 5  # [NEW] tối thiểu 5 mẫu mỗi class

    def _load_sample_with_min_per_class(self):
        if not os.path.exists(self.CSV_PATH):
            self.skipTest(f"Không tìm thấy file CSV: {self.CSV_PATH}")

        df = pd.read_csv(self.CSV_PATH, nrows=self.SAMPLE_ROWS)
        label_col = self.LABEL_COL if self.LABEL_COL in df.columns else df.columns[-1]

        # [NEW] Lọc class có đủ tối thiểu MIN_PER_CLASS mẫu
        vc = df[label_col].value_counts(dropna=False)
        keep_labels = vc[vc >= self.MIN_PER_CLASS].index
        df = df[df[label_col].isin(keep_labels)].copy()

        # Nếu sau lọc còn quá ít lớp hoặc quá ít mẫu -> skip
        if df.shape[0] < max(50, self.MIN_PER_CLASS * 2):
            self.skipTest(
                f"Sample sau lọc quá ít ({df.shape[0]} rows). "
                f"Hãy tăng SAMPLE_ROWS hoặc giảm MIN_PER_CLASS."
            )

        # Build X/y
        X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32, copy=True)
        y = df[label_col].to_numpy(copy=True)

        # [NEW] Guard: stratify yêu cầu mỗi lớp phải có >=2 mẫu, nhưng split có thể fail nếu quá ít
        # Với MIN_PER_CLASS=5 thì thường ok, nhưng vẫn thêm fallback.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        except ValueError:
            # Fallback: không stratify (smoke test chỉ cần không crash)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=None
            )

        # Normalize về [0, 1] bằng MinMaxScaler (fit trên train, transform cả train và test)
        # Clip để đảm bảo tất cả giá trị trong [0, 1]
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_train = np.clip(X_train, 0, 1)  # Đảm bảo [0, 1]
        X_test = scaler.transform(X_test)
        X_test = np.clip(X_test, 0, 1)  # Đảm bảo [0, 1]

        return X_train, X_test, y_train, y_test

    def test_pipeline_ver6_smoke(self):
        X_train, X_test, y_train, y_test = self._load_sample_with_min_per_class()

        pipeline = HedgeAlgebraPipeline(
            n_clusters=6,
            center_init="ver6",
            optimize_parameters=False,
            use_information_gain=False,
            test_size=0.2,
            random_state=42,
            log_level="ERROR",
            log_to_file=False,
        )

        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "accuracy"))
        self.assertTrue(hasattr(result, "cluster_centers"))

    def test_pipeline_legacy_smoke(self):
        X_train, X_test, y_train, y_test = self._load_sample_with_min_per_class()

        pipeline = HedgeAlgebraPipeline(
            n_clusters=6,
            center_init="legacy",
            optimize_parameters=False,
            use_information_gain=False,
            test_size=0.2,
            random_state=42,
            log_level="ERROR",
            log_to_file=False,
        )

        result = pipeline.run_with_data(X_train, X_test, y_train, y_test)
        self.assertIsNotNone(result)

    def test_autocluster_smoke(self):
        X_train, _, _, _ = self._load_sample_with_min_per_class()

        auto = AutoClusterPipeline(
            min_clusters=2,
            max_clusters=4,     # test nhanh
            optimize_params=False,
            center_init="ver6",
            log_level="ERROR",
        )

        auto_result = auto.run(X_train, selection_metric="silhouette")
        self.assertIsNotNone(auto_result)
        self.assertTrue(hasattr(auto_result, "best_n_clusters"))


if __name__ == "__main__":
    unittest.main()
