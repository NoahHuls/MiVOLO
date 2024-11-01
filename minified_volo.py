import logging
import cv2
import torch
from mivolo.data.data_reader import InputType, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)
_logger = logging.getLogger("inference")

def run_inference(input_path, detector_weights, checkpoint, device="cpu"):
    setup_default_logging()
    if torch.cuda.is_available() and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    predictor = Predictor({
        "detector_weights": detector_weights,
        "checkpoint": checkpoint,
        "device": device,
        "with_persons": True,
        "disable_faces": False,
        "draw": False
    }, verbose=False)

    if get_input_type(input_path) != InputType.Image:
        raise ValueError("Only image input is supported.")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image from {input_path}")
    
    detected_objects, _ = predictor.recognize(img)
    
    return detected_objects.ages if detected_objects else None

def get_age(input_path):
    return run_inference(
        input_path=input_path,
        detector_weights="MiVOLO/weights/yolov8x_person_face.pt",
        checkpoint="MiVOLO/weights/model_age_utk_4.23.pth.tar",
        device="cpu"
    )[1]

if __name__ == "__main__":
    age = get_age("FGNET/001A02.jpg")
    print(age)