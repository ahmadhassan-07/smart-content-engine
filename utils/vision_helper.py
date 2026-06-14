import cv2
import numpy as np
from PIL import Image


def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """
    PIL Image ko OpenCV BGR format mein convert karta hai.
    """
    rgb_array = np.array(pil_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def run_object_detection(cv_model, pil_image: Image.Image, confidence: float):
    """
    YOLOv8 model se object detection chalata hai.

    Returns:
        annotated_image (np.ndarray): Bounding boxes wali RGB image
        detected_objects (list): Detected object names ki list
        object_counts (dict): Har object ka count
    """
    opencv_image = pil_to_opencv(pil_image)

    # YOLO inference
    results = cv_model(opencv_image, conf=confidence, verbose=False)

    # Annotated frame (BGR) — plot() BGR return karta hai
    annotated_bgr = results[0].plot()

    # Display ke liye BGR → RGB
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Detected classes nikalo
    detected_objects = []
    object_counts = {}

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = results[0].names

        for cls_id in class_ids:
            name = class_names[cls_id]
            detected_objects.append(name)
            object_counts[name] = object_counts.get(name, 0) + 1

    return annotated_rgb, detected_objects, object_counts


def check_content_safety(detected_objects: list) -> tuple[bool, str]:
    """
    Detected objects ke basis par content safety check karta hai.

    Returns:
        is_safe (bool): True agar safe hai
        message (str): Safety status message
    """
    # Prohibited categories (extend kar sakte ho apni zaroorat ke mutabiq)
    prohibited_keywords = {
        "knife", "gun", "pistol", "rifle", "weapon",
        "scissors", "sword", "bomb", "explosive"
    }

    flagged = [obj for obj in detected_objects if obj.lower() in prohibited_keywords]

    if flagged:
        return False, f"⚠️ Prohibited items detected: {', '.join(set(flagged))}"
    return True, "✅ Content appears safe — no prohibited objects found."