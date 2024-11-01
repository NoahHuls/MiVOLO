import logging
import os

import cv2
import torch
import yt_dlp
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")

def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        if "url" in info_dict:
            direct_url = info_dict["url"]
            resolution = (info_dict["width"], info_dict["height"])
            fps = info_dict["fps"]
            yid = info_dict["id"]
            return direct_url, resolution, fps, yid

    return None, None, None, None

def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return res, fps

def run_inference(input_path, output_path, detector_weights, checkpoint, with_persons=False, disable_faces=False, draw=False, device="cuda"):
    setup_default_logging()
    if torch.cuda.is_available() and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(output_path, exist_ok=True)
    
    predictor = Predictor({
        "detector_weights": detector_weights,
        "checkpoint": checkpoint,
        "device": device,
        "with_persons": with_persons,
        "disable_faces": disable_faces,
        "draw": draw
    }, verbose=True)

    input_type = get_input_type(input_path)
    
    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not draw:
            raise ValueError("Video processing requires 'draw' to be set to True.")

        if "youtube" in input_path:
            direct_url, res, fps, yid = get_direct_video_url(input_path)
            if not direct_url:
                raise ValueError(f"Failed to get direct video URL for {input_path}")
            outfilename = os.path.join(output_path, f"out_{yid}.avi")
        else:
            bname = os.path.splitext(os.path.basename(input_path))[0]
            outfilename = os.path.join(output_path, f"out_{bname}.avi")
            res, fps = get_local_video_info(input_path)

        if draw:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving video output to {outfilename}.")

        for (detected_objects_history, frame) in predictor.recognize_video(input_path):
            if draw:
                out.write(frame)

    elif input_type == InputType.Image:
        image_files = get_all_files(input_path) if os.path.isdir(input_path) else [input_path]

        for img_p in image_files:
            img = cv2.imread(img_p)
            detected_objects, out_im = predictor.recognize(img)

            if draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(output_path, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")
    else:
        raise ValueError(f"Unsupported input type for {input_path}")

    return detected_objects.ages

def get_age(input_path):
        return run_inference(
        input_path=input_path,
        output_path="MiVOLO/PaperCode/output",
        detector_weights="MiVOLO/PaperCode/yolov8x_person_face.pt",
        checkpoint="MiVOLO/PaperCode/model_age_utk_4.23.pth.tar",
        draw=True,
        device="cpu"
    )[1]

if __name__ == "__main__":
    age = get_age("MiVOLO/PaperCode/WIN_20241101_13_29_05_Pro.jpg")
    print(age)


