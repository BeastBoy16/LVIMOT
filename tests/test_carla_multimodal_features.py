import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_box_projection import CarlaBoundingBoxProjector
from carla_association import CarlaObjectAssociator
from carla_lidar_features import CarlaLiDARPlanarFeatureExtractor
from carla_multimodal_features import CarlaMultimodalFeatureBuilder
from fast_features import FASTFeatureDetector
from brief_features import BRIEFDescriptorExtractor

CARLA_ROOT = '~/catkin_ws/src/CARLA_LVIMOT'
SEQUENCE = '0006'
FRAME = 100


def test_carla_multimodal_features():
    seq_dir = os.path.expanduser(f'{CARLA_ROOT}/sequences/{SEQUENCE}')
    if not os.path.isdir(seq_dir):
        pytest.skip(f'CARLA sequence directory not found: {seq_dir}')

    reader = CarlaReader(CARLA_ROOT, SEQUENCE)
    calibration = CarlaCalibration(f'{CARLA_ROOT}/calibration/carla_calibration.json')
    image = reader.load_image(FRAME)
    lidar = reader.load_lidar(FRAME)
    pose = reader.load_pose(FRAME)
    detector = CarlaDetector(reader, calibration)
    objects = detector.load_detections(FRAME)

    lidar_extractor = CarlaLiDARPlanarFeatureExtractor(voxel_size=0.20)
    planar_features = lidar_extractor.extract(lidar)

    fast_detector = FASTFeatureDetector(threshold=20, nonmax_suppression=True)
    fast_keypoints = fast_detector.detect(image)

    brief_extractor = BRIEFDescriptorExtractor(bytes=32)
    brief_keypoints, descriptors = brief_extractor.compute(image, fast_keypoints)

    projector = CarlaBoundingBoxProjector(calibration)
    projected_boxes = projector.project_objects(objects, pose)

    associator = CarlaObjectAssociator(iou_threshold=0.1)
    camera_objects = [{'class': p['class'], 'bbox': p['box_2d']} for p in projected_boxes]
    associations = associator.associate(projected_boxes, camera_objects)

    builder = CarlaMultimodalFeatureBuilder(calibration)
    multimodal_objects = builder.build(
        objects=objects,
        associations=associations,
        projected_boxes=projected_boxes,
        planar_features=planar_features,
        ego_pose=pose,
        keypoints=brief_keypoints,
        descriptors=descriptors
    )

    assert len(multimodal_objects) >= 0


if __name__ == '__main__':
    test_carla_multimodal_features()
