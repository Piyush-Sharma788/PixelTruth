import logging

import cv2
import numpy as np

from config import IMAGE_SIZE

logger = logging.getLogger(__name__)


def gradcam_available():
    return False


def get_backbone_submodel(_model):
    logger.warning("Grad-CAM is not supported for Hugging Face models (SigLIP/ViT).")
    return None


def find_last_conv_layer(_model):
    logger.warning("Grad-CAM is not supported for Hugging Face models (SigLIP/ViT).")
    return None


def make_gradcam_heatmap(img_array, _model, _last_conv_layer, pred_index=None):
    logger.warning("Grad-CAM is not supported for Hugging Face models (SigLIP/ViT).")
    return None


def overlay_heatmap(image, _heatmap, alpha=0.4):
    logger.warning("Grad-CAM is not supported for Hugging Face models (SigLIP/ViT).")
    return None
