import unittest

import numpy as np

from app.documents.passport_localization import detect_passport_page_padded, page_corners_from_mrz, page_corners_from_mrz_width, relative_to_mrz_width
from scripts.dataset.annotate_passport_mrz import relative_to_mrz


class PassportMrzAnnotationTests(unittest.TestCase):
    def test_page_coordinates_are_relative_to_mrz_polygon(self):
        mrz = np.float32([[10, 70], [110, 70], [110, 90], [10, 90]])
        page = np.float32([[0, 0], [120, 0], [120, 100], [0, 100]])
        relative = relative_to_mrz(mrz, page)
        self.assertTrue(np.allclose([[-0.1, -3.5], [1.1, -3.5], [1.1, 1.5], [-0.1, 1.5]], relative))
        self.assertTrue(np.allclose(page, page_corners_from_mrz(mrz, relative), atol=1e-5))
        self.assertTrue(np.allclose(page, page_corners_from_mrz_width(mrz, relative_to_mrz_width(mrz, page)), atol=1e-5))

    def test_padding_is_not_part_of_mrz_localization(self):
        image = np.zeros((120, 140, 3), dtype=np.uint8)
        relative = relative_to_mrz_width(np.float32([[20, 80], [120, 80], [120, 100], [20, 100]]), np.float32([[10, 10], [130, 10], [130, 110], [10, 110]]))
        result = detect_passport_page_padded(
            image,
            {"document_localization": {"strategy": "mrz_anchor", "page_corners_relative_to_mrz_width": relative}},
            lambda _image, **_: {"mrz_polygon": [[20, 80], [120, 80], [120, 100], [20, 100]]},
            10,
        )
        self.assertTrue(np.allclose([[20, 20], [140, 20], [140, 120], [20, 120]], result["corners"]))
