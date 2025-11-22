# tracking_with_localization_vpd_gps_modular.py

import cv2
import json
import numpy as np
import os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from math import radians, tan, cos
import time

from vakt_tracker import VAKTTracker   # Your tracker

# ----------------- GLOBAL CONSTANTS -----------------
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

LAUNCH_LAT = 42.94181399384941
LAUNCH_LON = 10.712085390159576

DRONE_ALTITUDE_M = 20.0
HFOV_DEG = 78.0
VFOV_DEG = 60.0

METHOD_NAME = "VPD-GPS"
# ----------------------------------------------------


# ================================================================
# Utility: Convert numpy types for JSON
# ================================================================
def to_native(obj):
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ================================================================
# GPS Projection
# ================================================================
def pixel_to_gps(cx, cy, width, height, lat, lon):
    """Project pixel to GPS coordinate using drone geometry."""
    earth = 6378137.0
    ground_w = 2 * DRONE_ALTITUDE_M * tan(radians(HFOV_DEG) / 2)
    ground_h = 2 * DRONE_ALTITUDE_M * tan(radians(VFOV_DEG) / 2)

    meters_x = ground_w / width
    meters_y = ground_h / height

    dx = (cx - width / 2) * meters_x
    dy = (cy - height / 2) * meters_y

    dLat = -dy / earth
    dLon = dx / (earth * cos(radians(lat)))

    return lat + dLat * 180 / np.pi, lon + dLon * 180 / np.pi


# ================================================================
# Remove swimmer inside boat
# ================================================================
def iou_area_relative(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    A = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    return inter / A


def remove_swimmers_inside_boats(det_boxes, det_labels, cat_map, threshold=0.6):
    boat_ids = [cid for cid, name in cat_map.items() if name.lower() == "boat"]
    swimmer_ids = [cid for cid, name in cat_map.items() if name.lower() in ("person", "swimmer")]

    fboxes, flabels = [], []

    for box, lbl in zip(det_boxes, det_labels):
        if lbl in swimmer_ids:
            inside = any(
                l in boat_ids and iou_area_relative(box, b) >= threshold
                for b, l in zip(det_boxes, det_labels)
            )
            if inside:
                continue

        fboxes.append(box)
        flabels.append(lbl)

    return fboxes, flabels


# ================================================================
# Draw tracked boxes
# ================================================================
def draw_tracked_boxes(frame, tracks, cat_id_to_name, localization_points=None):
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for idx, t in enumerate(tracks):
        bbox = t["bbox"]
        tid = t["id"]
        cid = t["class_id"]

        class_name = cat_id_to_name.get(cid, "unknown").lower()
        if class_name == "person": class_name = "swimmer"
        color = (255, 0, 0) if class_name == "swimmer" else (0, 0, 255)

        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, y1 - 15), f"ID:{tid} {class_name}", fill=(255, 255, 255), font=font)

        # GPS marker
        if localization_points:
            lat = localization_points[idx]["estimated_lat"]
            lon = localization_points[idx]["estimated_lon"]
            cx, cy = localization_points[idx]["center_pixel"]
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 255, 0))
            draw.text((cx + 5, cy - 10), f"{lat:.6f},{lon:.6f}", fill=(255, 255, 0), font=font)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


# ================================================================
# Save cropped snapshot (ONLY bounding box, rate-limited to 2 sec)
# ================================================================
def save_snapshot(frame, bbox, track_id, frame_idx, last_snapshot_time):
    x1, y1, x2, y2 = map(int, bbox)
    current = time.time()

    if current - last_snapshot_time.get(track_id, 0) < 2:
        return

    last_snapshot_time[track_id] = current

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return

    path = f"{SNAPSHOT_DIR}/swimmer_{track_id}_frame{frame_idx}.jpg"
    cv2.imwrite(path, crop)
    print(f"📸 Snapshot saved: {path}")


# ================================================================
# MAIN MODULE — RUN EVERYTHING
# ================================================================
def run_tracker(video_path):

    # Derive JSON automatically from naming or adjust here:
    # YOU must modify this if your JSON is not based on video name.
    coco_json_path = r"C:\Users\Lenovo-Z50-70\Desktop\Project I - Drone Vision\dataset\mobdrone_annotation\mobdroneannotations.json"
    output_video_path = "final_output.mp4"
    localization_output_path = "localization_results.json"

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Load COCO annotations
    with open(coco_json_path) as f:
        coco = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Match video frames to annotation frames
    video_file = os.path.splitext(os.path.basename(video_path))[0]
    imgs = [img for img in coco["images"] if video_file in img["file_name"]]
    anns = [a for a in coco["annotations"] if a["image_id"] in {i["id"] for i in imgs}]
    img_to_anns = defaultdict(list)
    for a in anns:
        img_to_anns[a["image_id"]].append(a)

    tracker = VAKTTracker(max_age=1000, min_hits=3, iou_threshold=0.3, appearance_lambda=0.7)

    localization_results = []
    frame_idx = 0
    last_snapshot_time = {}

    # ----------------- MAIN LOOP -----------------
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(imgs):
            break

        img_id = imgs[frame_idx]["id"]

        det_boxes, det_labels = [], []

        for ann in img_to_anns[img_id]:
            x, y, w, h = ann["bbox"]
            det_boxes.append([x, y, x + w, y + h])
            det_labels.append(ann["category_id"])

        det_boxes, det_labels = remove_swimmers_inside_boats(det_boxes, det_labels, id_to_name)

        tracked = tracker.update(np.array(det_boxes), np.array(det_labels), np.ones(len(det_boxes)), frame)[0]

        frame_data = []

        for t in tracked:
            bbox = t["bbox"]
            tid = t["id"]
            cid = t["class_id"]

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            lat, lon = pixel_to_gps(cx, cy, width, height, LAUNCH_LAT, LAUNCH_LON)

            class_name = id_to_name[cid].lower()

            # Save swimmer snapshot ONLY once per 2 seconds
            if class_name in ("person", "swimmer"):
                save_snapshot(frame, bbox, tid, frame_idx, last_snapshot_time)

            frame_data.append({
                "track_id": int(tid),
                "class_id": int(cid),
                "bbox": list(map(float, bbox)),
                "estimated_lat": lat,
                "estimated_lon": lon,
                "center_pixel": [cx, cy],
                "method": METHOD_NAME
            })

        out.write(draw_tracked_boxes(frame.copy(), tracked, id_to_name, frame_data))
        localization_results.append(frame_data)

        frame_idx += 1

    # ----------------- SAVE RESULTS -----------------
    cap.release()
    out.release()

    with open(localization_output_path, "w") as f:
        json.dump(to_native(localization_results), f, indent=2)

    print("\n✅ Tracking Complete!")
    print(f"🎥 Video Saved: {output_video_path}")
    print(f"📄 GPS Output Saved: {localization_output_path}")
    print(f"🖼 Snapshots Folder: {SNAPSHOT_DIR}")


# ================================================================
# RUNNING POINT
# ================================================================
if __name__ == "__main__":
    run_tracker(
        r"D:\Dataset\video_split_fullhd\DJI_0915_0018_60m.mp4"
    )
