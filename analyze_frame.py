import cv2
import numpy as np
from ultralytics import YOLO

def analyze_frame(frame, model, conf_threshold=0.3, draw_skeleton=True):
    """
    Runs fighter detection on a frame and returns a frame with optional skeleton/highlighted points.

    Parameters
    ----------
    frame : np.ndarray
        BGR image from OpenCV
    model : YOLO model
        YOLOv8-pose model
    conf_threshold : float
        YOLO confidence threshold
    draw_skeleton : bool
        Whether to draw the structural skeleton and keypoints

    Returns
    -------
    output : np.ndarray
        Frame with bounding boxes and optionally skeleton/keypoints
    detections : list
        List of detected persons with bounding boxes and keypoints
    """

    results = model(frame, conf=conf_threshold, verbose=False)[0]
    output = frame.copy()
    detections = []

    if results.keypoints is None:
        return output, detections

    # Define COCO structural skeleton: shoulders, elbows, wrists, hips, knees, ankles
    skeleton_structural = [
        (5, 6),      # shoulders
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10), # right arm
        (5, 11), (6, 12), # torso sides
        (11, 12),         # hips
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    for box, kpts in zip(results.boxes, results.keypoints):
        if int(box.cls) != 0:
            continue

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if draw_skeleton:
            # Convert keypoints and confidence to NumPy arrays
            xy_array = kpts.xy.cpu().numpy().reshape(-1, 2)      # shape (num_joints, 2)
            conf_array = kpts.conf.cpu().numpy().flatten()        # shape (num_joints,)

            # Draw keypoints
            for i in range(xy_array.shape[0]):
                x, y = xy_array[i]
                conf = conf_array[i]
                if conf > 0.7:
                    cv2.circle(output, (int(x), int(y)), 3, (0, 0, 255), -1)

            # Draw structural skeleton lines
            for start_idx, end_idx in skeleton_structural:
                if conf_array[start_idx] > 0.7 and conf_array[end_idx] > 0.7:
                    x0, y0 = xy_array[start_idx]
                    x1, y1 = xy_array[end_idx]
                    cv2.line(output, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

        # Store detection info
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "keypoints": kpts
        })

    return output, detections


