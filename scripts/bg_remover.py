from PIL import Image
from rembg import remove, new_session
import argparse
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from person_detector import detect_person
from birefnet_infer import birefnet_remove_background


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

    # Create transparent canvas and paste
    transparent_canvas = Image.new("RGBA", input.size, (0, 0, 0, 0))
    transparent_canvas.paste(result, (x_min, y_min), result)
    transparent_canvas.save(output_path)

    print(f"U2Net: Background removed and saved to: {output_path}")


def remove_background_sam(input_img, localize_person=False):
    input_path = os.path.join(RAW_DIR, input_img)
    output_path = os.path.join(PROCESSED_DIR, input_img.split('.')[0] + '_sam.png')

    print("SAM: Loading image...")
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

    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    main_mask = sorted_masks[0]['segmentation']

    cropped_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
    cropped_rgba[~main_mask] = (0, 0, 0, 0)

    full_canvas = np.zeros((h, w, 4), dtype=np.uint8)
    full_canvas[y_offset:y_offset + cropped_rgba.shape[0], x_offset:x_offset + cropped_rgba.shape[1]] = cropped_rgba

    cv2.imwrite(output_path, full_canvas)
    print(f"SAM: Background removed and saved to: {output_path}")

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

    print("RMBG: Loading image...")
    orig_image = Image.open(input_path).convert("RGB")
    orig_im_np = np.array(orig_image)

    if localize_person:
        print("RMBG: Detecting person with DETR...")
        person_bbox = detect_person(input_path)
        if not person_bbox:
            print("RMBG: No person detected. Exiting.")
            return
        x_min, y_min, x_max, y_max = person_bbox
        cropped_np = orig_im_np[y_min:y_max, x_min:x_max]
    else:
        print("RMBG: Skipping person localization.")
        cropped_np = orig_im_np
        x_min, y_min = 0, 0
        y_max, x_max = orig_im_np.shape[0], orig_im_np.shape[1]

    print("RMBG: Loading model...")
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(DEVICE)
    model.eval()

    model_input_size = [1024, 1024]
    image_tensor = preprocess_image(cropped_np, model_input_size).to(DEVICE)

    with torch.no_grad():
        result = model(image_tensor)

    print("RMBG: Postprocessing mask...")
    result_mask = postprocess_image(result[0][0], cropped_np.shape[:2])
    pil_mask = Image.fromarray(result_mask).resize((x_max - x_min, y_max - y_min))

    print("RMBG: Creating transparent canvas...")
    transparent_canvas = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    cropped_rgba = Image.fromarray(cropped_np).convert("RGBA")
    cropped_rgba.putalpha(pil_mask)

    transparent_canvas.paste(cropped_rgba, (x_min, y_min), mask=cropped_rgba)

    transparent_canvas.save(output_path)
    print(f"RMBG: Background removed and saved to: {output_path}")

def remove_background_birefnet(input_img, localize_person=False):
    input_path = os.path.join(RAW_DIR, input_img)

    if localize_person:
        print("Localizing person...")
        bbox = detect_person(input_path)
        if not bbox:
            print("No person detected.")
            return
    else:
        bbox = None

    print("Running BiRefNet...")
    birefnet_remove_background(input_path, PROCESSED_DIR, person_bbox=bbox)
    print("Done.")

if __name__ == '__main__':
    all_models = ['u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use', 'sam', 'rmbg', 'birefnet']
    parser = argparse.ArgumentParser(description='Remove background from an image.')
    parser.add_argument('input', help='Name of the input image file in the data/raw directory')
    parser.add_argument('--model', choices=all_models, default='u2net', help='Model to use for background removal')
    parser.add_argument('--localize_person', action='store_true', help='Use DETR to localize person before removing background (only for RMBG)')
    args = parser.parse_args()

    input_file_path = os.path.join(RAW_DIR, args.input)
    if not os.path.exists(input_file_path):
        print(f"Input file not found: {input_file_path}")
        exit(1)

    if args.model == 'sam':
        remove_background_sam(args.input, localize_person=args.localize_person)
    elif args.model == 'rmbg':
        remove_background_rmbg(args.input, localize_person=args.localize_person)
    elif args.model == "birefnet":
        remove_background_birefnet(args.input, args.localize_person)
    else:
        remove_background_u2net(args.input, args.model, localize_person=args.localize_person)