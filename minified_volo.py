import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    age_estimation = run_inference(
        input_path=input_path,
        detector_weights="MiVOLO/weights/yolov8x_person_face.pt",
        checkpoint="MiVOLO/weights/model_age_utk_4.23.pth.tar",
        device="cpu"
    )

    if len(age_estimation) > 0:
        for age in age_estimation:
            if age is not None:
                return age
    return None

if __name__ == "__main__":
    data = []
    fails = []
    foldername = "FGNET"

    for filename in os.listdir(foldername):
        if filename.endswith(".JPG"):
            original_age = int(filename[4:6])

            image_path = os.path.join(foldername, filename.replace("JPG", "jpg"))
            estimated_age = get_age(image_path)
            if estimated_age is not None:
                data.append({
                    "Filename": filename,
                    "Original Age": original_age,
                    "Estimated Age": round(estimated_age)
                })
            else:
                fails.append(image_path)

    df = pd.DataFrame(data)

    mae = np.mean(np.abs(df["Original Age"] - df["Estimated Age"]))
    print(f"Mean Absolute Error (MAE): {mae}")

    df['Original_Class'] = np.where(df['Original Age'] >= 25, 'Over 25', 'Under 25')
    df['Estimated_Class'] = np.where(df['Estimated Age'] >= 25, 'Over 25', 'Under 25')

    true_positive = ((df['Original_Class'] == 'Over 25') & (df['Estimated_Class'] == 'Over 25')).sum()
    true_negative = ((df['Original_Class'] == 'Under 25') & (df['Estimated_Class'] == 'Under 25')).sum()
    false_positive = ((df['Original_Class'] == 'Under 25') & (df['Estimated_Class'] == 'Over 25')).sum()
    false_negative = ((df['Original_Class'] == 'Over 25') & (df['Estimated_Class'] == 'Under 25')).sum()

    confusion_matrix = pd.DataFrame({
        "Predicted": ["Over 25", "Under 25"],
        "Over 25": [true_positive, false_positive],
        "Under 25": [false_negative, true_negative]
    }).set_index("Predicted")

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Over/Under 25)')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    print(df)
    plt.show()