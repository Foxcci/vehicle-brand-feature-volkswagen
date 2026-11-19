"""Numerical regression against the supplied notebook and analytical properties."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gm11_forecast import gm11_predict_next, resample_contour, display_curve
from euclidean_distance import calculate_tables
from vgg_gradcam import background_mask
from PIL import Image

spec = importlib.util.spec_from_file_location("original1124", ROOT / "legacy/original_1124_core.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class ScientificRegressionTests(unittest.TestCase):
    def test_original_formula_parity(self):
        for values in ([120, 125, 132, 139, 148], [250, 220, 203, 194, 185],
                       [75, 83, 92, 103, 112]):
            self.assertEqual(gm11_predict_next(values, "legacy"), original.gm11_predict_next(values))
            self.assertAlmostEqual(gm11_predict_next(values, "stable"), original.gm11_predict_next(values), places=8)

    def test_constant_series_stable_limit(self):
        self.assertEqual(gm11_predict_next([175]*5, "stable"), 175)

    def test_stable_near_constant_series(self):
        self.assertAlmostEqual(gm11_predict_next([100, 100+1e-6, 100+2e-6, 100+3e-6, 100+4e-6], "stable"),
                               100+5e-6, places=6)

    def test_resampling_matches_original(self):
        polygon = np.array([[30., 40.], [220., 50.], [250., 170.], [80., 180.]])
        np.testing.assert_allclose(resample_contour(polygon), original.resample_contour(polygon), atol=1e-12)

    def test_smoothing_does_not_mutate_forecast(self):
        angle = np.linspace(0, 2*np.pi, 100, endpoint=False)
        points = np.column_stack([170+100*np.cos(angle), 170+50*np.sin(angle)])
        before = points.copy()
        for mode in ("none", "periodic", "light"):
            displayed = display_curve(points, mode)
            self.assertTrue(np.isfinite(displayed).all())
            np.testing.assert_array_equal(before, points)

    def test_known_distances_sd_and_ranking(self):
        features = {("A", "LA"): np.array([0., 0.], np.float32),
                    ("A", "PA"): np.array([3., 0.], np.float32),
                    ("A", "SA"): np.array([0., 4.], np.float32),
                    ("B", "LA"): np.array([0., 0.], np.float32),
                    ("B", "PA"): np.array([6., 0.], np.float32),
                    ("B", "SA"): np.array([0., 8.], np.float32)}
        rows, pairs = calculate_tables(features, parts=("A", "B"))
        self.assertEqual(len(pairs), 3)
        self.assertEqual(rows[0]["mu"], 4)
        self.assertEqual(rows[0]["sigma"], 1)
        self.assertEqual(rows[0]["CV"], .25)
        self.assertEqual(rows[0]["rank_CV"], 1.5)
        self.assertAlmostEqual(rows[0]["S"], .7*1+.3*1.5)

    def test_mask_retains_enclosed_gray_region(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        image[5:27, 5:27] = 255
        image[12:20, 12:20] = 128
        mask, color = background_mask(Image.fromarray(image))
        self.assertEqual(color, [128, 128, 128])
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[15, 15], 1)


if __name__ == "__main__":
    unittest.main()
