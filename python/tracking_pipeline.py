"""
This file implements the main tracking pipeline.
We instantiate a Tracker object to maintain state across frames.
"""

import cv2
from dataclasses import dataclass
from python.detect import detect_people_bboxes
from python.annotate import annotate_bbox
from python.identify import identify

@dataclass
class Tracker:
    curr_frame = None
    curr_bboxes: list = None
    curr_bbox_ids: dict = None

tracker = Tracker()

def tracking_pipeline(next_frame):
    global tracker

    if next_frame is None:
        return None

    # Initialize
    if tracker.curr_frame is None:
        tracker.curr_frame = next_frame.copy()
        tracker.curr_bboxes = detect_people_bboxes(next_frame)

        tracker.curr_bbox_ids = identify(next_frame, [], tracker.curr_bboxes)

    else:
        # Next frames
        next_bboxes = detect_people_bboxes(next_frame)
        next_bbox_ids = identify(next_frame, tracker.curr_bboxes, next_bboxes)

        tracker.curr_frame = next_frame.copy()
        tracker.curr_bboxes = next_bboxes
        tracker.curr_bbox_ids = next_bbox_ids

    # Prepare bbox data for annotation
    frame_bboxes = []
    if tracker.curr_bboxes:
        for i, bbox in enumerate(tracker.curr_bboxes):
            bbox_id = tracker.curr_bbox_ids.get(i, i)
            x1, y1, x2, y2, conf = bbox
            frame_bboxes.append({
                "id": int(bbox_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf)
            })

    # Annotate for display
    final_frame = next_frame.copy()
    if tracker.curr_bboxes:
        for i, bbox in enumerate(tracker.curr_bboxes):
            bbox_id = tracker.curr_bbox_ids.get(i, i)
            final_frame = annotate_bbox(final_frame, bbox, label=str(bbox_id))

    return final_frame
