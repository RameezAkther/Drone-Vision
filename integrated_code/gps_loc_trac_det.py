# tracking_with_rtdetr_localization.py
import os
import time
import json
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from math import radians, tan, cos

import torch
from torchvision import transforms
import torch.nn.functional as F

# --- change these imports if your RT-DETR repo uses different paths ---
from rtdetr_pytorch.config import get_config
from rtdetr_pytorch.modeling import build_model
# ----------------------------------------------------------------------

from vakt_tracker import VAKTTracker  # your tracker

# ----------------- GLOBALS / DEFAULTS -----------------
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

LAUNCH_LAT = 42.94181399384941
LAUNCH_LON = 10.712085390159576

DRONE_ALTITUDE_M = 20.0
HFOV_DEG = 78.0
VFOV_DEG = 60.0

METHOD_NAME = "VPD-GPS"

# detection thresholds
SCORE_THR = 0.5   # keep detections > this score
# -----------------------------------------------------


def to_native(obj):
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------- Geometric helpers ----------------------
def pixel_to_gps(cx, cy, width, height, lat, lon):
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


def iou_area_relative(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    A = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    return inter / A


def remove_swimmers_inside_boats(det_boxes, det_labels, mapping, threshold=0.6):
    boat_ids = [cid for cid, name in mapping.items() if name.lower() == "boat"]
    swimmer_ids = [cid for cid, name in mapping.items() if name.lower() in ("person", "swimmer")]
    fboxes = []; flabels = []
    for box, lbl in zip(det_boxes, det_labels):
        if lbl in swimmer_ids:
            if any(l in boat_ids and iou_area_relative(box, b) >= threshold
                   for b, l in zip(det_boxes, det_labels)):
                continue
        fboxes.append(box); flabels.append(lbl)
    return fboxes, flabels
# ------------------------------------------------------------------


# ---------------------- Visualization / Snapshots ----------------------
def draw_tracked_boxes(frame, tracks, cat_id_to_name, localization_points=None):
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for idx, track in enumerate(tracks):
        bbox = track.get('bbox') if isinstance(track, dict) else track[0]
        track_id = track.get('id') if isinstance(track, dict) else track[1]
        class_id = track.get('class_id') if isinstance(track, dict) else track[2]

        if bbox is None: continue

        class_name = cat_id_to_name.get(class_id, f'Class-{class_id}').lower()
        if class_name == "person":
            class_name = "swimmer"; color = (255, 0, 0)
        elif class_name == "boat":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        x1, y1, x2, y2 = [float(x) for x in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 15), f"ID:{track_id} {class_name}", fill=(255, 255, 255), font=font)

        if localization_points and idx < len(localization_points):
            lat = localization_points[idx].get('estimated_lat')
            lon = localization_points[idx].get('estimated_lon')
            cx, cy = localization_points[idx]['center_pixel']
            draw.ellipse((cx-4, cy-4, cx+4, cy+4), fill=(255,255,0))
            draw.text((cx+5, cy-10), f"{lat:.6f},{lon:.6f}", fill=(255,255,0), font=font)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def save_snapshot(frame, bbox, track_id, frame_idx, last_snapshot_time, rate_seconds=2):
    x1, y1, x2, y2 = map(int, bbox)
    current = time.time()
    if current - last_snapshot_time.get(track_id, 0) < rate_seconds:
        return
    last_snapshot_time[track_id] = current
    # clip coords
    h, w = frame.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, w-1)
    y1, y2 = np.clip([y1, y2], 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    path = os.path.join(SNAPSHOT_DIR, f"swimmer_{track_id}_frame{frame_idx}.jpg")
    cv2.imwrite(path, crop)
    print(f"📸 Snapshot saved: {path}")
# -------------------------------------------------------------------------


# ---------------------- RT-DETR helper: load & detect ----------------------
def load_rtdetr_model(cfg_path, weights_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(cfg_path)
    model = build_model(cfg, is_train=False)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, device


# NOTE: This detection wrapper supports the common RT-DETR output:
#   - dict with 'pred_logits' (B,N,C) and 'pred_boxes' (B,N,4 in cxcywh normalized)
# If your model returns already-postprocessed results (list of dicts with 'boxes','scores','labels'),
# the function will use that format automatically.
def detect_rtdetr(model, device, frame, input_size=640, score_thr=SCORE_THR, max_detections=200):
    """
    frame: numpy BGR image (H,W,3)
    returns: boxes_xyxy (list of [x1,y1,x2,y2]), labels (list of int), scores (list of float)
    """
    # Preprocess: convert to RGB, resize to input_size, to tensor, normalize if possible
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # normalization - many RT-DETR implementations use ImageNet norm; keep it.
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensor = tf(img_pil).unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        outputs = model(tensor)

    # Case A: model returns list/dict of post-processed outputs (common for some wrappers)
    # e.g. outputs = [{'boxes':..., 'scores':..., 'labels':...}] or outputs[0]...
    if isinstance(outputs, (list, tuple)):
        first = outputs[0]
        if isinstance(first, dict) and ('boxes' in first and 'scores' in first):
            boxes = first['boxes'].cpu().numpy().tolist()
            scores = first['scores'].cpu().numpy().tolist()
            labels = first.get('labels', torch.zeros(len(boxes)).long()).cpu().numpy().tolist()
            # convert boxes if already in xyxy in pixels - assume they match original frame size
            return boxes, labels, scores

    if isinstance(outputs, dict) and ('pred_logits' in outputs and 'pred_boxes' in outputs):
        logits = outputs['pred_logits']  # (1,N,C)
        boxes = outputs['pred_boxes']    # (1,N,4) normalized cxcywh
    elif isinstance(outputs, (list, tuple)) and isinstance(outputs[0], dict) and ('pred_logits' in outputs[0]):
        logits = outputs[0]['pred_logits']
        boxes = outputs[0]['pred_boxes']
    else:
        # fallback: try to interpret outputs as postprocessed attribute
        # If nothing matches, try to return empty lists
        try:
            # sometimes model returns a Tensor of detections; attempt to parse
            return [], [], []
        except Exception:
            return [], [], []

    # logits: (1,N,C) where C = num_classes + 1 (background)
    probs = F.softmax(logits, dim=-1)[0]  # (N, C)
    scores, labels = probs[:, :-1].max(dim=-1)  # ignore last background class
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()  # class indices in [0..num_classes-1]
    boxes = boxes[0].cpu().numpy()  # (N,4) cxcywh normalized

    H, W = frame.shape[:2]
    boxes_xyxy = []
    selected_labels = []
    selected_scores = []
    for (cx, cy, w, h), lab, sc in zip(boxes, labels, scores):
        if sc < score_thr:
            continue
        # convert normalized cxcywh -> pixel xyxy relative to original frame size
        x_c = cx * W
        y_c = cy * H
        bw = w * W
        bh = h * H
        x1 = x_c - bw / 2
        y1 = y_c - bh / 2
        x2 = x_c + bw / 2
        y2 = y_c + bh / 2
        # clamp
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)
        boxes_xyxy.append([x1, y1, x2, y2])
        selected_labels.append(int(lab))
        selected_scores.append(float(sc))

    # optionally: keep only top-k by score
    if len(selected_scores) > max_detections:
        idxs = np.argsort(selected_scores)[-max_detections:]
        boxes_xyxy = [boxes_xyxy[i] for i in idxs]
        selected_labels = [selected_labels[i] for i in idxs]
        selected_scores = [selected_scores[i] for i in idxs]

    return boxes_xyxy, selected_labels, selected_scores
# ---------------------------------------------------------------------


# ---------------------- MAIN run function (modular) ----------------------
def run_tracker(video_path,
                rtdetr_cfg=None,
                rtdetr_weights=None,
                coco_json_path=None,
                output_video_path="final_tracked_video_with_localization_rtdetr.mp4",
                localization_output_path="localization_results_rtdetr.json"):
    """
    rtdetr_cfg, rtdetr_weights: path to RT-DETR config and trained weights.
    coco_json_path: optional - used to map class indices to names (if provided).
                   If not provided, class IDs will be used as numeric labels.
    """
    # load COCO categories if provided (for naming)
    id_to_name = {}
    if coco_json_path and os.path.exists(coco_json_path):
        with open(coco_json_path) as f:
            coco = json.load(f)
        id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        # NOTE: RT-DETR labels might be zero-based contiguous class indices;
        # mapping depends on how you trained the model. If your model used
        # COCO category indices as labels directly, you can use id_to_name.
    else:
        # fallback: numeric names only
        id_to_name = defaultdict(lambda: "class")

    # load model
    if rtdetr_cfg is None or rtdetr_weights is None:
        raise ValueError("Please provide rtdetr_cfg and rtdetr_weights paths.")
    model, device = load_rtdetr_model(rtdetr_cfg, rtdetr_weights)
    print(f"RT-DETR loaded on {device}")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # prepare images mapping if coco JSON supplied (not used for detection)
    imgs = []
    if coco_json_path and os.path.exists(coco_json_path):
        with open(coco_json_path) as f:
            coco = json.load(f)
        video_file = os.path.splitext(os.path.basename(video_path))[0]
        imgs = [img for img in coco["images"] if video_file in img["file_name"]]
    # build tracker
    tracker = VAKTTracker(max_age=1000, min_hits=3, iou_threshold=0.3, appearance_lambda=0.7)

    localization_results = []
    frame_index = 0
    last_snapshot_time = {}

    print("Starting detection+tracking loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect using RT-DETR
        det_boxes, det_labels, det_scores = detect_rtdetr(model, device, frame, input_size=640, score_thr=SCORE_THR)

        # If models were trained with 0-based class ids, you may want to map them to COCO ids.
        # Here we assume label indices map to COCO category indices if coco_json_path provided.
        # If a custom mapping exists, adapt here.
        if isinstance(id_to_name, dict) and len(id_to_name) > 0 and not isinstance(id_to_name, defaultdict):
            # try to map label index -> category id: if label indexes are consecutive starting 0,
            # we try to build a mapping using sorted category ids. This is heuristic and can be changed.
            # For safety we try to use label as category id if present, else fallback to label+1
            mapped_labels = []
            for lab in det_labels:
                # direct match?
                if lab in id_to_name:
                    mapped_labels.append(lab)
                elif (lab + 1) in id_to_name:
                    mapped_labels.append(lab + 1)
                else:
                    mapped_labels.append(lab)  # leave numeric
            det_labels = mapped_labels

        # Remove swimmers inside boats (same logic as before)
        det_boxes, det_labels = remove_swimmers_inside_boats(det_boxes, det_labels, id_to_name)

        # If no detections, still update tracker with empty arrays (tracker handles)
        if len(det_boxes) == 0:
            tracked = tracker.update(np.array([]), np.array([]), np.array([]), frame)[0]
        else:
            tracked = tracker.update(np.array(det_boxes), np.array(det_labels), np.ones(len(det_boxes)), frame)[0]

        frame_loc = []
        swimmer_ids = []; boat_ids = []
        for t in tracked:
            bbox = t["bbox"]; tid = t["id"]; cid = t["class_id"]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            lat, lon = pixel_to_gps(cx, cy, width, height, LAUNCH_LAT, LAUNCH_LON)

            cname = id_to_name.get(cid, str(cid)).lower() if isinstance(id_to_name, dict) else str(cid)

            if cname in ("person", "swimmer"):
                swimmer_ids.append(tid)
                save_snapshot(frame, bbox, tid, frame_index, last_snapshot_time, rate_seconds=2)
            if cname == "boat":
                boat_ids.append(tid)

            frame_loc.append({
                "track_id": int(tid),
                "class_id": int(cid),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "estimated_lat": lat,
                "estimated_lon": lon,
                "center_pixel": [cx, cy],
                "method": METHOD_NAME
            })

        # Print status
        os.system('cls' if os.name == 'nt' else 'clear')
        print("===== LIVE TRACKING STATUS =====")
        print(f"FRAME: {frame_index}")
        print(f"SWIMMERS: {swimmer_ids}   (count: {len(swimmer_ids)})")
        print(f"BOATS:    {boat_ids}      (count: {len(boat_ids)})")
        print("----------------------------------")
        for obj in frame_loc:
            print(f"ID:{obj['track_id']}  LAT:{obj['estimated_lat']:.6f}  LON:{obj['estimated_lon']:.6f}")
        print("\n(Warnings and snapshots will appear below as events...)")

        out.write(draw_tracked_boxes(frame.copy(), tracked, id_to_name, frame_loc))
        localization_results.append(frame_loc)
        frame_index += 1

    cap.release()
    out.release()

    with open(localization_output_path, "w") as f:
        json.dump(to_native(localization_results), f, indent=2)

    print("\n✅ DONE — Output saved.")
    print(f"🎥 Video: {output_video_path}")
    print(f"📄 JSON:  {localization_output_path}")
    print(f"🖼 Snapshots saved in: {SNAPSHOT_DIR}")


# ---------------------- CLI / example usage ----------------------
if __name__ == "__main__":
    # EDIT these paths to your files:
    VIDEO_PATH = r"D:\Dataset\video_split_fullhd\DJI_0915_0018_60m.mp4"
    RTDETR_CFG = "configs/rtdetr/enhanced_rtdetr_convnext_fsg.yml"   # change to your trained config if needed
    RTDETR_WEIGHTS = "weights/enhanced_rtdetr_trained.pth"          # path to your trained weights
    COCO_JSON = r"C:\Users\Lenovo-Z50-70\Desktop\Project I - Drone Vision\dataset\mobdrone_annotation\mobdroneannotations.json"

    run_tracker(VIDEO_PATH, rtdetr_cfg=RTDETR_CFG, rtdetr_weights=RTDETR_WEIGHTS, coco_json_path=COCO_JSON)
