import unittest
import numpy as np

from src.clustering import HedgeAlgebraClustering


class TestHedgeAlgebraClustering(unittest.TestCase):
    def setUp(self):
        self.clustering = HedgeAlgebraClustering(
            n_clusters=3, theta=0.5, alpha=0.5,
            center_init="legacy", log_level="ERROR"
        )

    def test_initialize_cluster_centers_legacy(self):
        centers = self.clustering.initialize_cluster_centers()
        expected = np.array([0.125, 0.5, 0.625])
        np.testing.assert_allclose(centers, expected)

    def test_update_cluster_centers_signature(self):
        semantic_values = np.array([0.1, 0.2, 0.5, 0.9])
        cluster_labels = np.array([1, 1, 2, 3])
        old_centers = [0.0, 0.0, 0.0]
        new_centers = self.clustering.update_cluster_centers(
            semantic_values, cluster_labels, old_centers
        )
        np.testing.assert_allclose(new_centers, [0.15, 0.5, 0.9])


    def test_ver6_init_mode(self):
        ver6_model = HedgeAlgebraClustering(
            n_clusters=3, theta=0.5, alpha=0.5, center_init="ver6", log_level="ERROR"
        )
        centers = ver6_model.initialize_cluster_centers()
        np.testing.assert_allclose(centers, [0.25, 0.5, 0.75])


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
