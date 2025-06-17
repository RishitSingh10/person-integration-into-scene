from PIL import Image
from rembg import remove, new_session
import argparse
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.transforms.functional import normalize
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.person_detector import detect_person
from utils.birefnet_infer import birefnet_remove_background

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Constants
SAM_CHECKPOINT = os.path.join(MODELS_DIR, "sam_vit_h_4b8939.pth")
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def remove_background_u2net(input_img, model_name='isnet-general-use', localize_person=False):
    input_path = os.path.join(RAW_DIR, input_img)
    output_path = os.path.join(PROCESSED_DIR, input_img.split('.')[0] + '.png')
    input = Image.open(input_path).convert("RGB")

    if localize_person:
        print("U2Net: Detecting person with DETR...")
        person_bbox = detect_person(input_path)
        if not person_bbox:
            print("U2Net: No person detected. Exiting.")
            return
        x_min, y_min, x_max, y_max = person_bbox
        cropped = input.crop((x_min, y_min, x_max, y_max))
    else:
        print("U2Net: Skipping localization.")
        x_min, y_min = 0, 0
        x_max, y_max = input.size
        cropped = input

    session = new_session(model_name)
    result = remove(cropped, session=session, post_process_mask=True, alpha_matting=True)

    transparent_canvas = Image.new("RGBA", input.size, (0, 0, 0, 0))
    transparent_canvas.paste(result, (x_min, y_min), result)
    transparent_canvas.save(output_path)
    print(f"U2Net: Saved to: {output_path}")


def remove_background_sam(input_img, localize_person=False):
    input_path = os.path.join(RAW_DIR, input_img)
    output_path = os.path.join(PROCESSED_DIR, input_img.split('.')[0] + '_sam.png')
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f"Error: Could not read image from {input_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    if localize_person:
        print("SAM: Detecting person with DETR...")
        person_bbox = detect_person(input_path)
        if not person_bbox:
            print("SAM: No person detected. Exiting.")
            return
        x_min, y_min, x_max, y_max = person_bbox
        image_rgb = image_rgb[y_min:y_max, x_min:x_max]
        x_offset, y_offset = x_min, y_min
    else:
        x_offset, y_offset = 0, 0

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"Missing SAM checkpoint: {SAM_CHECKPOINT}")
        return

    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    masks = mask_generator.generate(image_rgb)
    if not masks:
        print("SAM: No objects detected.")
        return

    main_mask = sorted(masks, key=(lambda x: x['area']), reverse=True)[0]['segmentation']
    cropped_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
    cropped_rgba[~main_mask] = (0, 0, 0, 0)

    full_canvas = np.zeros((h, w, 4), dtype=np.uint8)
    full_canvas[y_offset:y_offset + cropped_rgba.shape[0], x_offset:x_offset + cropped_rgba.shape[1]] = cropped_rgba

    cv2.imwrite(output_path, full_canvas)
    print(f"SAM: Saved to: {output_path}")


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=model_input_size, mode='bilinear')
    image = im_tensor / 255.0
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma, mi = torch.max(result), torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    return np.squeeze(im_array)


def remove_background_rmbg(input_img, localize_person=False):
    input_path = os.path.join(RAW_DIR, input_img)
    output_path = os.path.join(PROCESSED_DIR, input_img.split('.')[0] + '_rmbg.png')

    orig_image = Image.open(input_path).convert("RGB")
    orig_im_np = np.array(orig_image)

    if localize_person:
        bbox = detect_person(input_path)
        if not bbox:
            print("RMBG: No person detected.")
            return
        x_min, y_min, x_max, y_max = bbox
        cropped_np = orig_im_np[y_min:y_max, x_min:x_max]
    else:
        cropped_np = orig_im_np
        x_min, y_min = 0, 0
        y_max, x_max = orig_im_np.shape[0], orig_im_np.shape[1]

    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(DEVICE)
    model.eval()
    model_input_size = [1024, 1024]
    image_tensor = preprocess_image(cropped_np, model_input_size).to(DEVICE)

    with torch.no_grad():
        result = model(image_tensor)

    result_mask = postprocess_image(result[0][0], cropped_np.shape[:2])
    pil_mask = Image.fromarray(result_mask).resize((x_max - x_min, y_max - y_min))

    transparent_canvas = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    cropped_rgba = Image.fromarray(cropped_np).convert("RGBA")
    cropped_rgba.putalpha(pil_mask)

    transparent_canvas.paste(cropped_rgba, (x_min, y_min), mask=cropped_rgba)
    transparent_canvas.save(output_path)
    print(f"RMBG: Saved to: {output_path}")

def remove_background_rmbg2(input_img, localize_person=False):
    input_path = os.path.join(RAW_DIR, input_img)
    output_path = os.path.join(PROCESSED_DIR, input_img.split('.')[0] + '_rmbg2.png')
    image = Image.open(input_path).convert("RGB")
    original_size = image.size  # Save original size before cropping

    if localize_person:
        bbox = detect_person(input_path)
        if not bbox:
            print("RMBG 2.0: No person detected.")
            return
        image = image.crop(bbox)
        offset = (bbox[0], bbox[1])
    else:
        offset = (0, 0)

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True).to(DEVICE)
    model.eval()
    input_tensor = transform_image(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred).resize(image.size)
    image.putalpha(pred_pil)

    # Always use original size for the final canvas
    final_canvas = Image.new("RGBA", original_size, (0, 0, 0, 0))
    final_canvas.paste(image, offset, image)
    final_canvas.save(output_path)
    print(f"RMBG 2.0: Saved to: {output_path}")


def remove_background_birefnet(input_img, localize_person=False):
    input_path = os.path.join(RAW_DIR, input_img)
    bbox = detect_person(input_path) if localize_person else None
    birefnet_remove_background(input_path, PROCESSED_DIR, person_bbox=bbox)
    print("BiRefNet: Done.")


if __name__ == '__main__':
    all_models = ['u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use', 'sam', 'rmbg', 'rmbg2', 'birefnet']
    parser = argparse.ArgumentParser(description='Remove background from an image.')
    parser.add_argument('input', help='Name of the input image file in the data/raw directory')
    parser.add_argument('--model', choices=all_models, default='u2net', help='Model to use for background removal')
    parser.add_argument('--localize_person', action='store_true', help='Use DETR to localize person before removing background')
    args = parser.parse_args()

    input_file_path = os.path.join(RAW_DIR, args.input)
    if not os.path.exists(input_file_path):
        print(f"Input file not found: {input_file_path}")
        exit(1)

    if args.model == 'sam':
        remove_background_sam(args.input, args.localize_person)
    elif args.model == 'rmbg':
        remove_background_rmbg(args.input, args.localize_person)
    elif args.model == 'rmbg2':
        remove_background_rmbg2(args.input, args.localize_person)
    elif args.model == 'birefnet':
        remove_background_birefnet(args.input, args.localize_person)
    else:
        remove_background_u2net(args.input, args.model, args.localize_person)