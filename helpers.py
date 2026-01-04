import cv2
import numpy as np

def draw_bbox(output, box, color=(0, 255, 0), thickness=2):
    """Draw bounding box on the frame."""
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
    return output, (x1, y1, x2, y2)

def draw_keypoints(output, xy_array, conf_array, conf_thresh=0.9, color=(0, 0, 255), radius=3):
    """Draw keypoints as circles."""
    for i in range(xy_array.shape[0]):
        x, y = xy_array[i]
        conf = conf_array[i]
        if conf > conf_thresh:
            cv2.circle(output, (int(x), int(y)), radius, color, -1)
    return output

def draw_skeleton(output, xy_array, conf_array, skeleton, conf_thresh=0.9, color=(255, 0, 0), thickness=2):
    """Draw structural skeleton lines between keypoints."""
    for start_idx, end_idx in skeleton:
        if conf_array[start_idx] > conf_thresh and conf_array[end_idx] > conf_thresh:
            x0, y0 = xy_array[start_idx]
            x1, y1 = xy_array[end_idx]
            cv2.line(output, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
    return output
