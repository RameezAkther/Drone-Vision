# tracking.py (Corrected for Pillow's TypeError)
import cv2
import json
import numpy as np
import os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# Make sure 'vakt_tracker.py' is in the same directory
from vakt_tracker import VAKTTracker

# ======== Paths (replace these) ========
video_path = "distress_signal_1_20m.mp4"
coco_json_path = "mobdroneannotations.json"
output_video_path = "final_tracked_video4.mp4"
video_name_identifier = "DJI_0804_0016_20m"
# ======================================


# --- Helper function for drawing ---
def draw_tracked_boxes(frame, tracks, cat_id_to_name):
    """Draws tracked bounding boxes with IDs on the frame."""
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for track in tracks:
        # This robustly handles the tracker's output format
        if isinstance(track, dict):
            bbox = track.get('bbox')
            track_id = track.get('id')
            class_id = track.get('class_id')
        else: # Fallback for an unexpected list format from the tracker
            try:
                bbox = track[0]
                track_id = track[1]
                class_id = track[2]
            except (IndexError, TypeError):
                # If the format is still wrong, skip this track to prevent a crash
                print(f"Warning: Skipping malformed track object: {track}")
                continue
        
        if bbox is None or track_id is None or class_id is None:
            continue
        
        class_name = cat_id_to_name.get(class_id, f'Class-{class_id}').lower()
        
        if class_name == "person":
            class_name = "swimmer"
            color = (255, 0, 0)  # Red
        elif class_name == "boat":
            color = (0, 0, 255)  # Blue
        else:
            color = (0, 255, 0)  # Green for others
        
        label = f"ID:{track_id} {class_name.capitalize()}"
        
        # --- THE FIX: Convert bbox from numpy array to a list for the drawing function ---
        draw.rectangle(list(bbox), outline=color, width=3)
        # --- END OF FIX ---
        
        text_size = font.getbbox(label)
        text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
        text_bg = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
        draw.rectangle(text_bg, fill=color)
        draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# 1. Load COCO JSON data
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Build class map dynamically
cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# 2. Filter images and annotations for the specific video
print(f"Filtering annotations for video identifier: {video_name_identifier}...")
filtered_images = [img for img in coco_data["images"] if video_name_identifier in img["file_name"]]
filtered_image_ids = {img["id"] for img in filtered_images}
filtered_annotations = [ann for ann in coco_data.get("annotations", []) if ann["image_id"] in filtered_image_ids]
print(f"Found {len(filtered_images)} frames and {len(filtered_annotations)} annotations for this video.")

# 3. Build a mapping for easy lookup
image_annotations = defaultdict(list)
for ann in filtered_annotations:
    image_annotations[ann["image_id"]].append(ann)

# 4. Open video for processing
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- Main processing loop with tracker ---
tracker = VAKTTracker(max_age=1000, min_hits=3, iou_threshold=0.3, appearance_lambda=0.7)
sorted_images = sorted(filtered_images, key=lambda x: x["file_name"])
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_index >= len(sorted_images):
        break

    image_id = sorted_images[frame_index]["id"]
    gt_anns_for_frame = image_annotations.get(image_id, [])

    det_boxes, det_scores, det_labels = [], [], []
    for ann in gt_anns_for_frame:
        x, y, w, h = ann['bbox']
        det_boxes.append([x, y, x + w, y + h])
        det_scores.append(1.0)
        det_labels.append(ann['category_id'])
    
    # This correctly handles the tracker's tuple output by taking the first element [0]
    tracked_objects = tracker.update(
        np.array(det_boxes),
        np.array(det_labels),
        np.array(det_scores),
        frame
    )[0] 
    
    # Pass the dynamically created map to the drawing function
    output_frame = draw_tracked_boxes(frame.copy(), tracked_objects, cat_id_to_name)
    
    out.write(output_frame)
    print(f"Processing frame {frame_index + 1}/{len(sorted_images)}...", end='\r')
    frame_index += 1

cap.release()
out.release()
print(f"\n✅ Done! Tracked video saved to: {output_video_path}")

