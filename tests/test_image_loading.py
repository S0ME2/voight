import unittest

import cv2


class ImageLoadingTests(unittest.TestCase):
    def test_missing_image_returns_none(self):
        self.assertIsNone(cv2.imread("./pic/MyID.jpg"))
