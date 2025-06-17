from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

print("Initializing DETR Person Detector...")
PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
MODEL = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL.to(DEVICE)
print(f"DETR model moved to {DEVICE}.")


def detect_person(image_path):
    """
    Detects objects in an image using DETR and returns the bounding box
    of the person with the highest confidence score.

    Args:
        image_path (str): The path to the input image.

    Returns:
        tuple: A bounding box (x_min, y_min, x_max, y_max) for the most
               confident person detection, or None if no person is found.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

    print("Localizing person using DETR model...")

    inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = PROCESSOR.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]

    best_person = None

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if MODEL.config.id2label[label.item()] == 'person':
            if best_person is None or score > best_person['score']:
                best_person = {
                    'score': score.item(),
                    'box': [int(round(i)) for i in box.tolist()]
                }

    if best_person:
        box = best_person['box']
        print(f"DETR: Detected person with confidence {best_person['score']:.3f} at location {box}")
        return tuple(box)
    else:
        print("DETR: No person detected in the image.")
        return None


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        detected_box = detect_person(image_file)
        if detected_box:
            print(f"\nFinal Bounding Box: {detected_box}")
    else:
        print("\nUsage: python person_detector.py <path_to_image>")