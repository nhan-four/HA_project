"""
File: codebase_project/tests/test_theory_init_modes.py

Regression Tests for BOTH init modes:
- center_init="ver6"  : Theo mapping Ver 6 (C-, theta, C+) + heuristic VC/LC (implementation convention)
- center_init="legacy": Theo công thức legacy đang triển khai trong src/clustering.py

Các test được chia rõ:
A) "Theory-locked" (những gì Ver 6 định nghĩa tường minh): N=2, N=3, midpoint rule, L2 squared objective
B) "Implementation-locked" (legacy + heuristic VC/LC): expected values theo code hiện tại để chống regression
"""

import unittest
import numpy as np

from src.clustering import HedgeAlgebraClustering, ParameterOptimizer


class TestInitModes(unittest.TestCase):
    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _make_1d_X(values: np.ndarray) -> np.ndarray:
        """Tạo X dạng (n,1) để semantic_values = mean(X, axis=1) == values."""
        return np.asarray(values, dtype=float).reshape(-1, 1)

    # -------------------------
    # FAIL-FAST / VALIDATION
    # -------------------------
    def test_invalid_center_init_fail_fast(self):
        with self.assertRaises(ValueError):
            HedgeAlgebraClustering(center_init="wrong")

        with self.assertRaises(ValueError):
            ParameterOptimizer(center_init="wrong")

    # -------------------------
    # A) THEORY-LOCKED (Ver 6 tường minh)
    # -------------------------
    def test_theory_n2_centers_formula_both_modes(self):
        """
        Ver 6 (N=2) định nghĩa tường minh:
          C- = θ(1-α)
          C+ = θ + α(1-θ)

        Công thức này áp dụng giống nhau cho cả ver6 & legacy trong code hiện tại.
        """
        theta = 0.5
        alpha = 0.5

        expected = [
            theta * (1 - alpha),          # 0.25
            theta + alpha * (1 - theta),  # 0.75
        ]

        for mode in ("ver6", "legacy"):
            model = HedgeAlgebraClustering(
                n_clusters=2, theta=theta, alpha=alpha, center_init=mode, log_level="ERROR"
            )
            centers = model.initialize_cluster_centers()
            self.assertEqual(len(centers), 2)
            np.testing.assert_allclose(centers, expected, rtol=0, atol=1e-12)

    def test_theory_n3_centers_ver6_mapping(self):
        """
        Ver 6 (N=3) mapping tường minh:
          [C-, θ, C+]
        """
        theta = 0.5
        alpha = 0.5

        model = HedgeAlgebraClustering(
            n_clusters=3, theta=theta, alpha=alpha, center_init="ver6", log_level="ERROR"
        )
        centers = model.initialize_cluster_centers()

        expected = [
            theta * (1 - alpha),          # 0.25
            theta,                        # 0.5
            theta + alpha * (1 - theta),  # 0.75
        ]
        np.testing.assert_allclose(centers, expected, rtol=0, atol=1e-12)

    def test_midpoint_rule_left_inclusive_both_modes(self):
        """
        Midpoint thresholding:
          x <= midpoint -> cụm trái
        Code dùng searchsorted(side='left') nên phải đúng cho cả ver6/legacy.
        """
        semantic_values = np.array([0.49, 0.50, 0.51], dtype=float)
        centers = [0.0, 1.0]  # midpoint = 0.5
        expected = np.array([1, 1, 2], dtype=np.int32)

        for mode in ("ver6", "legacy"):
            model = HedgeAlgebraClustering(n_clusters=2, center_init=mode, log_level="ERROR")
            labels = model.assign_to_clusters(semantic_values, centers)
            np.testing.assert_array_equal(labels, expected)
            self.assertEqual(labels.dtype, np.int32)

    def test_objective_l2_squared_both_modes(self):
        """
        Objective (SSZ) theo Ver 6:
          SSZ = Sum((Sd_i - SC_k)^2) -> min

        Không phụ thuộc init mode.
        """
        semantic_values = np.array([0.2, 0.6, 0.0, 1.0, 0.4], dtype=float)
        cluster_centers = [0.25, 0.75]
        cluster_labels = np.array([1, 2, 1, 2, 1], dtype=np.int32)

        # expected total = 0.1725 (tính tay)
        expected_total = 0.1725

        for mode in ("ver6", "legacy"):
            opt = ParameterOptimizer(center_init=mode, log_level="ERROR")
            total = opt.calculate_total_distance(semantic_values, cluster_labels, cluster_centers)
            self.assertAlmostEqual(total, expected_total, places=12)

    def test_end_to_end_pdf_example_n2_ver6(self):
        """
        End-to-end theo ví dụ semantic values (N=2), ver6.
        Dùng X 1D để semantic_values = mean(X)=values.
        """
        semantic_values = np.array([0.2, 0.6, 0.0, 1.0, 0.4], dtype=float)
        X = self._make_1d_X(semantic_values)

        model = HedgeAlgebraClustering(
            n_clusters=2, theta=0.5, alpha=0.5, center_init="ver6", log_level="ERROR"
        )
        result = model.fit(X)

        expected_labels = np.array([1, 2, 1, 2, 1], dtype=np.int32)
        np.testing.assert_array_equal(result.cluster_labels, expected_labels)

        # hội tụ về [0.2, 0.8]
        expected_centers = [0.2, 0.8]
        np.testing.assert_allclose(result.cluster_centers, expected_centers, rtol=0, atol=1e-12)

        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.n_iterations, 1)

    # -------------------------
    # B) IMPLEMENTATION-LOCKED (regression theo code hiện tại)
    # -------------------------
    def test_legacy_n3_centers_match_current_code(self):
        """
        Legacy (theo src/clustering.py hiện tại):
          N=3: [ θ(1-α)^2, θ, θ + α(1-θ)(1-α) ]
        Với θ=0.5, α=0.5 -> [0.125, 0.5, 0.625]
        """
        theta = 0.5
        alpha = 0.5

        model = HedgeAlgebraClustering(
            n_clusters=3, theta=theta, alpha=alpha, center_init="legacy", log_level="ERROR"
        )
        centers = model.initialize_cluster_centers()
        expected = [0.125, 0.5, 0.625]
        np.testing.assert_allclose(centers, expected, rtol=0, atol=1e-12)

    def test_ver6_n6_centers_match_current_heuristic(self):
        """
        Ver6 N=6 theo heuristic trong code (implementation convention):
          VC- = θ(1-α)^2
          C-  = θ(1-α)
          LC- = θ(1-0.5α)
          LC+ = θ + 0.5α(1-θ)
          C+  = θ + α(1-θ)
          VC+ = θ + α(1-θ)(1+α)
        Với θ=0.5, α=0.5:
          VC- = 0.125
          C-  = 0.25
          LC- = 0.375
          LC+ = 0.625
          C+  = 0.75
          VC+ = 0.875
        """
        theta = 0.5
        alpha = 0.5

        model = HedgeAlgebraClustering(
            n_clusters=6, theta=theta, alpha=alpha, center_init="ver6", log_level="ERROR"
        )
        centers = model.initialize_cluster_centers()
        expected = [0.125, 0.25, 0.375, 0.625, 0.75, 0.875]
        np.testing.assert_allclose(centers, expected, rtol=0, atol=1e-12)

    def test_legacy_n6_centers_match_current_code(self):
        """
        Legacy N=6 theo src/clustering.py hiện tại:
          [
            θ(1-α),
            θ(1-α/2),
            θ(1-α/4),
            θ + α/4(1-θ),
            θ + α/2(1-θ),
            θ + α(1-θ)
          ]
        Với θ=0.5, α=0.5:
          [0.25, 0.375, 0.4375, 0.5625, 0.625, 0.75]
        """
        theta = 0.5
        alpha = 0.5

        model = HedgeAlgebraClustering(
            n_clusters=6, theta=theta, alpha=alpha, center_init="legacy", log_level="ERROR"
        )
        centers = model.initialize_cluster_centers()
        expected = [0.25, 0.375, 0.4375, 0.5625, 0.625, 0.75]
        np.testing.assert_allclose(centers, expected, rtol=0, atol=1e-12)

    def test_end_to_end_legacy_n3_small_dataset(self):
        """
        End-to-end regression cho legacy N=3 (dataset nhỏ, determinism cao).
        Dùng semantic values để dễ kiểm soát.
        """
        semantic_values = np.array([0.1, 0.2, 0.6, 0.7, 0.9], dtype=float)
        X = self._make_1d_X(semantic_values)

        model = HedgeAlgebraClustering(
            n_clusters=3, theta=0.5, alpha=0.5, center_init="legacy", log_level="ERROR"
        )
        result = model.fit(X)

        # Kỳ vọng hội tụ về:
        # cluster1: [0.1, 0.2] -> 0.15
        # cluster2: [0.6, 0.7] -> 0.65   (0.7 nằm boundary sẽ về trái vì side='left')
        # cluster3: [0.9]      -> 0.9
        expected_centers = [0.15, 0.65, 0.9]
        np.testing.assert_allclose(result.cluster_centers, expected_centers, rtol=0, atol=1e-12)

        expected_labels = np.array([1, 1, 2, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(result.cluster_labels, expected_labels)

        self.assertTrue(result.converged)


if __name__ == "__main__":
    unittest.main()
