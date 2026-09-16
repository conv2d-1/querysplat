import cv2
import numpy as np


def filtered_points_with_confidence(points, confidence, h, w, output_conf_ratio=None, output_conf_thresh=None, colors=None):
    if confidence.shape[0] != h or confidence.shape[1] != w:
        confidence = cv2.resize(
            confidence,
            dsize=(w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    confidence = confidence.reshape(-1)

    if output_conf_ratio is not None:
        conf_thresh = np.percentile(confidence, output_conf_ratio * 100)
        confidence_mask = confidence > conf_thresh
    else:
        confidence_mask = confidence > output_conf_thresh

    filtered_points = points[confidence_mask]
    filtered_colors = colors[confidence_mask] if colors is not None else None

    return filtered_points, filtered_colors
