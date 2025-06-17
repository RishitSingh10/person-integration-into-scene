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

def birefnet_remove_background(input_path, output_dir, person_bbox=None):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    filename = os.path.splitext(os.path.basename(input_path))[0]

    if person_bbox:
        x_min, y_min, x_max, y_max = person_bbox
        image = image.crop((x_min, y_min, x_max, y_max))

    input_tensor = transform_image(image).unsqueeze(0).to(DEVICE).half()

    with torch.no_grad():
        pred = BiRefNetModel(input_tensor)[-1].sigmoid().cpu()[0].squeeze()

    pred_mask = transforms.ToPILImage()(pred).resize(image.size)
    masked_image = refine_foreground(image, pred_mask)
    masked_image.putalpha(pred_mask)

    pred_mask.save(os.path.join(output_dir, f"{filename}_mask.png"))
    masked_image.save(os.path.join(output_dir, f"{filename}_subject.png"))

    return os.path.join(output_dir, f"{filename}_subject.png")