import torch
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForImageSegmentation
from BiRefNet.image_proc import refine_foreground

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model once
print("BiRefNet: Loading model...")
BiRefNetModel = AutoModelForImageSegmentation.from_pretrained(
    'zhengpeng7/BiRefNet', trust_remote_code=True
)
BiRefNetModel.to(DEVICE).eval().half()
print("BiRefNet: Model ready.")

# Image transform
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import os
from PIL import Image
import torch
from torchvision import transforms

def birefnet_remove_background(input_path, output_dir, person_bbox=None):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    original_size = image.size  # Save original size before any cropping
    filename = os.path.splitext(os.path.basename(input_path))[0]

    if person_bbox:
        cropped_image = image.crop(person_bbox)
        offset = (person_bbox[0], person_bbox[1])
        image_to_process = cropped_image
    else:
        offset = (0, 0)
        image_to_process = image

    input_tensor = transform_image(image_to_process).unsqueeze(0).to(DEVICE).half()

    with torch.no_grad():
        pred = BiRefNetModel(input_tensor)[-1].sigmoid().cpu()[0].squeeze()

    pred_mask = transforms.ToPILImage()(pred).resize(image_to_process.size)
    masked_image = refine_foreground(image_to_process, pred_mask)
    masked_image.putalpha(pred_mask)

    # Final output image should match original dimensions
    final_canvas = Image.new("RGBA", original_size, (0, 0, 0, 0))

    # Paste masked cropped image at the correct offset
    final_canvas.paste(masked_image, offset, masked_image)

    # Save mask and result
    pred_mask.save(os.path.join(output_dir, f"{filename}_mask.png"))
    output_path = os.path.join(output_dir, f"{filename}_subject.png")
    final_canvas.save(output_path)

    return output_path
