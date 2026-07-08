import os
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from Neural import (
    DigitRecognizerModel,
    displayed_brush_to_pixels,
    preprocess_digit_image,
    stratified_train_val_split,
)


class DigitRecognizerModelTests(unittest.TestCase):
    def _tiny_data(self):
        X = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        y = np.array([0, 1, 1], dtype=int)
        return X, y

    def test_fit_uses_remainder_batch_when_batch_size_exceeds_dataset(self):
        X, y = self._tiny_data()
        model = DigitRecognizerModel(input_dim=4, hidden_dim1=5, hidden_dim2=3, output_dim=2, dropout_rate=0.0)
        model.initialize_weights()
        before = model.W1.copy()

        model.fit(X, y, X[:1], y[:1], epochs=1, batch_size=10)

        self.assertFalse(np.allclose(before, model.W1))

    def test_loaded_model_can_continue_training(self):
        X, y = self._tiny_data()
        model = DigitRecognizerModel(input_dim=4, hidden_dim1=5, hidden_dim2=3, output_dim=2, dropout_rate=0.0)
        model.fit(X, y, X[:1], y[:1], epochs=1, batch_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npz")
            model.save_weights(path)

            loaded = DigitRecognizerModel()
            loaded.load_weights(path)
            loaded.fit(X, y, X[:1], y[:1], epochs=1, batch_size=2)

        predictions = loaded.predict(X)
        self.assertEqual(predictions.shape, (len(X),))

    def test_one_hot_rejects_out_of_range_labels(self):
        with self.assertRaises(ValueError):
            DigitRecognizerModel.one_hot_encode(np.array([0, 10]), num_classes=10)

    def test_preprocess_digit_image_centers_drawn_ink(self):
        image = Image.new("L", (200, 200), color=0)
        draw = ImageDraw.Draw(image)
        draw.line((120, 30, 130, 170), fill=255, width=12)

        processed = preprocess_digit_image(image).reshape(32, 32)
        rows, cols = np.where(processed > 0.05)

        self.assertGreater(processed.max(), 0.5)
        self.assertGreaterEqual(rows.min(), 1)
        self.assertLessEqual(rows.max(), 30)
        self.assertGreaterEqual(cols.min(), 1)
        self.assertLessEqual(cols.max(), 30)
        self.assertAlmostEqual((rows.min() + rows.max()) / 2, 15.5, delta=1)
        self.assertAlmostEqual((cols.min() + cols.max()) / 2, 15.5, delta=1)

    def test_preprocess_digit_image_rejects_blank_canvas(self):
        image = Image.new("L", (200, 200), color=0)

        with self.assertRaises(ValueError):
            preprocess_digit_image(image)

    def test_stratified_split_keeps_every_digit_in_both_sets(self):
        X = np.arange(60 * 4).reshape(60, 4)
        y = np.repeat(np.arange(10), 6)

        X_train, y_train, X_val, y_val = stratified_train_val_split(X, y)

        self.assertEqual(set(y_train), set(range(10)))
        self.assertEqual(set(y_val), set(range(10)))
        self.assertEqual(len(X_train) + len(X_val), len(X))

    def test_displayed_brush_scale_maps_to_requested_pixel_range(self):
        self.assertEqual(displayed_brush_to_pixels(1), 15)
        self.assertEqual(displayed_brush_to_pixels(15), 30)
        self.assertGreater(displayed_brush_to_pixels(8), 15)
        self.assertLess(displayed_brush_to_pixels(8), 30)


if __name__ == "__main__":
    unittest.main()
