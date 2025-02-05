import json
import numpy as np
import cv2
import os
from pycocotools import mask as coco_mask
from collections import defaultdict

# Configuration
COCO_JSON_PATH = "/workspace/new/test/_annotations.coco.json"  # Path to COCO format JSON file
OUTPUT_DIR = "/workspace/new/masks"               # Directory to save masks

def create_segmentation_masks():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load COCO data
    with open(COCO_JSON_PATH, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    # Group annotations by image ID
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann['image_id']].append(ann)
    
    # Process each image
    for img_id, anns in img_anns.items():
        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Initialize mask (use uint16 to handle large category IDs)
        mask = np.zeros((img_height, img_width), dtype=np.uint16)
        
        # Process each annotation
        for ann in anns:
            category_id = 1 if ann['category_id'] in [4, 5] else 0
            seg = ann['segmentation']
            iscrowd = ann.get('iscrowd', 0)
            
            if not seg:  # Skip if no segmentation data
                continue
            
            try:
                if isinstance(seg, dict) or iscrowd == 1:
                    # Handle RLE format
                    rle = coco_mask.frPyObjects(seg, img_height, img_width)
                    binary_mask = coco_mask.decode(rle)
                    mask[binary_mask == 1] = category_id
                else:
                    # Handle polygon format
                    contours = []
                    for polygon in seg:
                        np_poly = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                        contours.append(np_poly.reshape((-1, 1, 2)))
                    cv2.fillPoly(mask, contours, color=category_id)
            except Exception as e:
                print(f"Error processing annotation {ann['id']} in image {img_id}: {str(e)}")
                continue
        
        # Convert to optimal data type
        max_id = mask.max()
        if max_id == 0:
            mask = mask.astype(np.uint8)
        else:
            mask = mask.astype(np.uint16 if max_id > 255 else np.uint8)
        
        # Save mask
        base_name = os.path.splitext(img_info['file_name'])[0]
        mask_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask for image {img_id} to {mask_path}")

if __name__ == "__main__":
    create_segmentation_masks()
