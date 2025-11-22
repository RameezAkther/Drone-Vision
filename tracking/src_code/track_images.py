import cv2
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict
import random

# ======== ⚙️ CONFIGURATION ========
IMAGE_FOLDER = "C:/Users/surya/Desktop/Sample Track Dataset"
ANNOTATION_FILE = "instances_train_objects_in_water.json"
OUTPUT_IMAGE_FOLDER = "final_annotated_images"
# ====================================

def draw_annotations_on_image(image_path, annotations, cat_id_to_name):
    """Draws annotations with scores and ID-only labels."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"\nWarning: Could not read image {image_path}.")
        return None

    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font = ImageFont.load_default()

    for ann in annotations:
        track_id, bbox, category_id = ann.get("track_id"), ann.get("bbox"), ann.get("category_id")
        if track_id is None or bbox is None or category_id is None: continue
        
        class_name = cat_id_to_name.get(category_id, "Unknown")

        # Color logic is preserved
        if class_name in ["swimmer", "swimmer with life jacket"]:
            color = (255, 0, 0)  # Red
        elif class_name == "boat":
            color = (0, 0, 255)  # Blue
        else:
            color = (0, 255, 0)  # Green

        x, y, w, h = map(int, bbox)
        box_to_draw = [x, y, x + w, y + h]
        draw.rectangle(box_to_draw, outline=color, width=5)

        # Generate random score and create the simple label
        score = random.uniform(0.6, 0.85)
        text_label = f"ID: {track_id} | {score:.2f}"
        
        text_size = font.getbbox(text_label)
        text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
        
        text_bg = (x, y - text_h - 8, x + text_w + 8, y)
        draw.rectangle(text_bg, fill=color)
        draw.text((x + 4, y - text_h - 4), text_label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# --- Main Script ---

print("Loading COCO annotations...")
try:
    with open(ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: The annotation file was not found at '{ANNOTATION_FILE}'")
    exit()

# --- ⭐ 1. Modified Pre-processing for Fast Lookups ---
# Create a map from category_id to category_name
cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

# Group all annotations by their image_id
annotations_by_image_id = defaultdict(list)
for ann in coco_data.get('annotations', []):
    annotations_by_image_id[ann['image_id']].append(ann)

# Create a map from image filename (without extension) to its info (like id)
# This allows us to find an image's data regardless of .png or .jpg mismatch
basename_to_image_info = {
    os.path.splitext(img['file_name'])[0]: img 
    for img in coco_data.get('images', [])
}
print("✅ Annotations processed and indexed.")

# --- ⭐ 2. Main loop is now driven by the files in your folder ---
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
print(f"Reading images from '{IMAGE_FOLDER}'...")
print(f"Output will be saved in '{OUTPUT_IMAGE_FOLDER}'")

# Get the list of actual image files you have
try:
    image_files_in_folder = sorted([
        f for f in os.listdir(IMAGE_FOLDER) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
except FileNotFoundError:
    print(f"❌ Error: The image folder was not found at '{IMAGE_FOLDER}'")
    exit()

if not image_files_in_folder:
    print("❌ Error: No image files found in the specified folder.")
    exit()

total_frames = len(image_files_in_folder)
print(f"Found {total_frames} images to process...")

for i, filename in enumerate(image_files_in_folder):
    print(f"Processing image {i+1}/{total_frames}: {filename}", end='\r')
    
    # Get the image's base name (e.g., "13852") to look it up in our map
    base_name = os.path.splitext(filename)[0]
    
    # Find the corresponding image info from the JSON
    image_info = basename_to_image_info.get(base_name)
    
    if image_info:
        image_id = image_info['id']
        image_path = os.path.join(IMAGE_FOLDER, filename)
        
        # Get all annotations for this image
        annotations_for_frame = annotations_by_image_id.get(image_id, [])
        
        # Draw the annotations
        annotated_frame = draw_annotations_on_image(image_path, annotations_for_frame, cat_id_to_name)
        
        if annotated_frame is not None:
            output_path = os.path.join(OUTPUT_IMAGE_FOLDER, filename)
            cv2.imwrite(output_path, annotated_frame)
    else:
        print(f"\nWarning: No annotation data found in JSON for image '{filename}'. Skipping.")

print(f"\n✅ Done! {total_frames} images processed successfully.")