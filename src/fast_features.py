try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np


class FASTFeatureDetector:
    def __init__(self, threshold=20, nonmax_suppression=True):
        self.threshold = threshold
        self.nonmax_suppression = nonmax_suppression
        if cv2 is not None and hasattr(cv2, 'FastFeatureDetector_create'):
            self.detector = cv2.FastFeatureDetector_create(
                threshold=self.threshold,
                nonmaxSuppression=self.nonmax_suppression
            )
        else:
            self.detector = None

    def detect(self, image):
        if image is None:
            raise ValueError('Input image is None.')
        if self.detector is None:
            return []
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return self.detector.detect(gray, None)

    def save_visualization(self, image, keypoints, output_file):
        if cv2 is not None:
            visualization = cv2.drawKeypoints(
                image, keypoints, None, color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.imwrite(output_file, visualization)
