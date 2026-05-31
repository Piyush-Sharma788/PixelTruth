import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import cv2

from preprocessing import detect_and_crop_face


def test_detect_and_crop_face_returns_original_if_no_face():
    # solid black image contains no faces
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped, face_box = detect_and_crop_face(image)
    
    assert face_box is None
    assert np.array_equal(cropped, image)


def test_detect_and_crop_face_selects_primary_face_mocked():
    mock_cascade = MagicMock()
    # Mock two faces detected:
    # Face 1: (10, 10, 30, 30) -> area 900
    # Face 2: (40, 40, 50, 50) -> area 2500 (primary)
    mock_cascade.detectMultiScale.return_value = np.array([
        [10, 10, 30, 30],
        [40, 40, 50, 50]
    ])
    mock_cascade.empty.return_value = False

    with patch("cv2.CascadeClassifier", return_value=mock_cascade), \
         patch("os.path.exists", return_value=True):
         
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Give some dummy color details to differentiate region
        image[40:90, 40:90, 0] = 255
        
        cropped, face_box = detect_and_crop_face(image)
        
        # Must select the primary (largest) face
        assert face_box == (40, 40, 50, 50)
        assert cropped.shape == (50, 50, 3)
        assert cropped[0, 0, 0] == 255
