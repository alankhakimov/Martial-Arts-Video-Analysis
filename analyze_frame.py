from helpers import*

def analyze_frame(frame, model, conf_threshold=0.3, draw_skeleton_flag=True):
    """
    Main analyze frame function
    """

    # analyze frame with YOLO
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    output = frame.copy()

    # list of dictionaries storing bounding boxes and keypoints
    detections = []

    if results.keypoints is None:
        return output, detections

    # Structural skeleton (COCO indices)
    skeleton_structural = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    # Bounding boxes, skeleton and key points
    for box, kpts in zip(results.boxes, results.keypoints):
        if int(box.cls) != 0:
            continue

        # Draw bounding box
        output, bbox_coords = draw_bbox(output, box)

        if draw_skeleton_flag:
            # Convert keypoints and confidence
            xy_array = kpts.xy.cpu().numpy().reshape(-1, 2)
            conf_array = kpts.conf.cpu().numpy().flatten()

            # Draw keypoints and skeleton
            output = draw_keypoints(output, xy_array, conf_array, conf_thresh=0.9)
            output = draw_skeleton(output, xy_array, conf_array, skeleton_structural, conf_thresh=0.9)

        # Store detection info
        detections.append({
            "bbox": bbox_coords,
            "keypoints": kpts
        })

    return output, detections



