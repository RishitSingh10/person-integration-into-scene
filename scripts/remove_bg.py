from PIL import Image
from rembg import remove, new_session
import argparse
import os
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# SAM Model Configuration
SAM_CHECKPOINT = os.path.join(MODELS_DIR, "sam_vit_h_4b8939.pth")
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def remove_background_u2net(input_img, model_name='isnet-general-use'):
    input_path = os.path.join(RAW_DIR, input_img)
    output_path = os.path.join(PROCESSED_DIR, input_img.split('.')[0] + '.png')

    input = Image.open(input_path)

    model_name = model_name
    session = new_session(model_name)
    output = remove(input, session=session, post_process_mask=True, alpha_matting=True)
    output.save(output_path)

    print(f"Background removed and saved to: {output_path}")

def remove_background_sam(input_img):
    """Removes background using Meta's Segment Anything Model (SAM)."""
    input_path = os.path.join(RAW_DIR, input_img)
    output_path = os.path.join(PROCESSED_DIR, input_img.split('.')[0] + '_sam.png')

    print("SAM: Loading image...")
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f"Error: Could not read image from {input_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("SAM: Loading model... (This may take a moment)")
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"Error: SAM checkpoint not found at {SAM_CHECKPOINT}")
        print("Please download it from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return

    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model=sam)

    print("SAM: Generating masks...")
    masks = mask_generator.generate(image_rgb)

    if not masks:
        print("SAM: No objects detected.")
        return

    # Sort masks by area and select the largest one
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    main_mask = sorted_masks[0]['segmentation']

    # Create a transparent background image
    h, w = main_mask.shape
    output_image = np.zeros((h, w, 4), dtype=np.uint8)

    # Apply the mask to the original image
    output_image[main_mask] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)[main_mask]

    # Save the result
    cv2.imwrite(output_path, output_image)
    print(f"SAM: Background removed and saved to: {output_path}")

if __name__ == '__main__':
    all_models = ['u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use', 'sam']
    parser = argparse.ArgumentParser(description='Remove background from an image.')
    parser.add_argument('input', help='Name of the input image file in the data/raw directory')
    parser.add_argument('--model', choices=all_models, default='u2net', help='Model to use for background removal')

    args = parser.parse_args()

    input_file_path = os.path.join(RAW_DIR, args.input)
    if not os.path.exists(input_file_path):
        print(f"Input file not found: {input_file_path}")
        exit(1)

    if args.model == 'sam':
        remove_background_sam(args.input)
    else:
        remove_background_u2net(args.input, args.model)