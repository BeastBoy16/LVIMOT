try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np


class BRIEFDescriptorExtractor:
    """BRIEF descriptor extractor with a deterministic NumPy fallback.

    OpenCV moved BRIEF into opencv-contrib. The fallback keeps the project
    functional even when xfeatures2d is unavailable instead of silently
    returning zero visual descriptors.
    """

    def __init__(self, bytes=32, patch_radius=15, seed=1337):
        self.bytes = int(bytes)
        self.patch_radius = int(patch_radius)
        self.num_bits = self.bytes * 8
        if (
            cv2 is not None
            and hasattr(cv2, "xfeatures2d")
            and hasattr(cv2.xfeatures2d, "BriefDescriptorExtractor_create")
        ):
            self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=self.bytes)
        else:
            self.extractor = None

        rng = np.random.default_rng(seed)
        # Gaussian BRIEF pattern concentrated near the keypoint.
        sigma = max(1.0, self.patch_radius / 2.5)
        pairs = rng.normal(0.0, sigma, size=(self.num_bits, 4))
        pairs = np.rint(pairs).astype(np.int32)
        pairs = np.clip(pairs, -self.patch_radius, self.patch_radius)
        self.pattern = pairs

    def _fallback(self, gray, keypoints):
        if cv2 is None:
            return [], None
        h, w = gray.shape[:2]
        kept = []
        descriptors = []
        r = self.patch_radius
        p = self.pattern
        for kp in keypoints:
            x, y = kp.pt
            xi, yi = int(round(x)), int(round(y))
            if xi - r < 0 or yi - r < 0 or xi + r >= w or yi + r >= h:
                continue
            a = gray[yi + p[:, 1], xi + p[:, 0]]
            b = gray[yi + p[:, 3], xi + p[:, 2]]
            bits = (a < b).astype(np.uint8)
            descriptors.append(np.packbits(bits))
            kept.append(kp)
        if not descriptors:
            return [], None
        return kept, np.asarray(descriptors, dtype=np.uint8)

    def compute(self, image, keypoints):
        if image is None:
            raise ValueError("Input image is None.")
        if len(keypoints) == 0:
            return [], None
        if cv2 is None:
            return [], None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if self.extractor is not None:
            kps, desc = self.extractor.compute(gray, keypoints)
            if desc is not None and len(kps) > 0:
                return kps, desc
        return self._fallback(gray, keypoints)
