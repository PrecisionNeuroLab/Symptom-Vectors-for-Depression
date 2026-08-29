import unittest
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import torch

from utils import *


class TestFiguresToHtml(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    def test_figures_to_html(self, mock_file):
        mock_fig = MagicMock()
        mock_fig.to_html.return_value = "<html><body>figure1</body></html>"
        figures_to_html([mock_fig], filename="test_dashboard.html")
        handle = mock_file()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn("figure1", written)
        self.assertTrue(written.startswith("<html>"))
        self.assertTrue(written.endswith("</body></html>\n"))


class TestCosDist(unittest.TestCase):
    def test_cos_dist_identical(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(cos_dist(a, b), 0.0)

    def test_cos_dist_opposite(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        self.assertAlmostEqual(cos_dist(a, b), 2.0)


class TestEuc(unittest.TestCase):
    def test_euc_distance(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([4.0, 6.0])
        self.assertAlmostEqual(euc(a, b), 5.0)

    def test_euc_zero(self):
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([0.0, 0.0])
        self.assertAlmostEqual(euc(a, b), 0.0)

    def test_euc_negative(self):
        a = torch.tensor([-1.0, -2.0])
        b = torch.tensor([-4.0, -6.0])
        self.assertAlmostEqual(euc(a, b), 5.0)


class TestMinMaxNormalizeRows(unittest.TestCase):
    def test_normalize_rows(self):
        df = pd.DataFrame([[1, 3], [2, 2]], index=["a", "b"], columns=["x", "y"])
        norm = min_max_normalize_rows(df)
        self.assertEqual(norm.loc["a", "x"], 0.0)
        self.assertEqual(norm.loc["a", "y"], 1.0)
        self.assertTrue(np.isnan(norm.loc["b", "x"]))
        self.assertTrue(np.isnan(norm.loc["b", "y"]))

    def test_all_equal_row(self):
        df = pd.DataFrame([[5, 5]], index=["a"], columns=["x", "y"])
        norm = min_max_normalize_rows(df)
        self.assertTrue(np.isnan(norm.loc["a", "x"]))
        self.assertTrue(np.isnan(norm.loc["a", "y"]))

    def test_negative_values(self):
        df = pd.DataFrame([[-2, 2]], index=["a"], columns=["x", "y"])
        norm = min_max_normalize_rows(df)
        self.assertEqual(norm.loc["a", "x"], 0.0)
        self.assertEqual(norm.loc["a", "y"], 1.0)

    def test_single_column(self):
        df = pd.DataFrame([[10], [20]], index=["a", "b"], columns=["x"])


class TestRemoveFirstVector(unittest.TestCase):
    def test_remove_first_vector(self):
        emb = torch.tensor([[1, 2], [3, 4], [5, 6]])
        result = remove_first_vector(emb)
        expected = torch.tensor([[3, 4], [5, 6]])
        self.assertTrue(torch.equal(result, expected))

    def test_single_row_tensor(self):
        emb = torch.tensor([[1, 2]])
        result = remove_first_vector(emb)
        expected = torch.empty((0, 2), dtype=emb.dtype)
        self.assertTrue(torch.equal(result, expected))

    def test_empty_tensor(self):
        emb = torch.empty((0, 2))
        result = remove_first_vector(emb)
        expected = torch.empty((0, 2))
        self.assertTrue(torch.equal(result, expected))


class TestCentroid(unittest.TestCase):
    def test_centroid_basic(self):
        emb = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected = torch.tensor([2.0, 3.0])
        result = centroid(emb)
        self.assertTrue(torch.allclose(result, expected))

    def test_centroid_single_vector(self):
        emb = torch.tensor([[5.0, 7.0]])
        expected = torch.tensor([5.0, 7.0])
        result = centroid(emb)
        self.assertTrue(torch.allclose(result, expected))


class TestMagnitude(unittest.TestCase):
    def test_magnitude_basic(self):
        emb = torch.tensor([3.0, 4.0])
        expected = 5.0  # sqrt(3^2 + 4^2)
        result = magnitude(emb)
        self.assertAlmostEqual(result, expected)

    def test_magnitude_zero(self):
        emb = torch.zeros(5)
        expected = 0.0
        result = magnitude(emb)
        self.assertAlmostEqual(result, expected)

    def test_magnitude_negative(self):
        emb = torch.tensor([-6.0, 8.0])
        expected = 10.0  # sqrt((-6)^2 + 8^2)
        result = magnitude(emb)
        self.assertAlmostEqual(result, expected)


class TestJaccardDistance(unittest.TestCase):
    def test_identical_sets(self):
        set1 = {1, 2, 3}
        set2 = {1, 2, 3}
        self.assertEqual(jaccard_distance(set1, set2), 0.0)

    def test_disjoint_sets(self):
        set1 = {1, 2}
        set2 = {3, 4}
        self.assertEqual(jaccard_distance(set1, set2), 1.0)

    def test_partial_overlap(self):
        set1 = {1, 2, 3}
        set2 = {2, 3, 4}
        # intersection = {2,3} (2), union = {1,2,3,4} (4)
        self.assertAlmostEqual(jaccard_distance(set1, set2), 1 - 2 / 4)

    def test_empty_sets(self):
        set1 = set()
        set2 = set()
        # By definition, intersection and union are 0, so should handle division by zero
        with self.assertRaises(ZeroDivisionError):
            jaccard_distance(set1, set2)


class TopEucNeighborsTest(unittest.TestCase):
    def test_sorted_ordering_and_distances(self):
        # Distances from query [1,0] are: 0, 1, 2, 3, 5 (unique and ascending)
        query = torch.tensor([1.0, 0.0])
        embeddings = torch.tensor([
            [1.0, 0.0],  # dist=0.0
            [2.0, 0.0],  # dist=1.0
            [-1.0, 0.0],  # dist=2.0
            [1.0, 3.0],  # dist=3.0
            [4.0, 4.0],  # dist=5.0
        ])
        result = list(top_euc_neighbors(query, embeddings))
        indices = [i for i, _ in result]
        dists = [d for _, d in result]

        self.assertEqual(indices, [0, 1, 2, 3, 4])
        self.assertAlmostEqual(dists[0], 0.0, places=6)
        self.assertAlmostEqual(dists[1], 1.0, places=6)
        self.assertAlmostEqual(dists[2], 2.0, places=6)
        self.assertAlmostEqual(dists[3], 3.0, places=6)
        self.assertAlmostEqual(dists[4], 5.0, places=6)

        self.assertIsInstance(result[0][0], int)
        self.assertIsInstance(result[0][1], float)

    def test_empty_embeddings(self):
        query = torch.tensor([1.0, 0.0])
        embeddings = torch.empty((0, 2))
        self.assertEqual(list(top_euc_neighbors(query, embeddings)), [])

    def test_invalid_shapes(self):
        query = torch.tensor([1.0, 0.0])
        embeddings_1d = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            list(top_euc_neighbors(query, embeddings_1d))

        query_2d = query.unsqueeze(0)
        embeddings_2d = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            list(top_euc_neighbors(query_2d, embeddings_2d))


if __name__ == "__main__":
    unittest.main()
