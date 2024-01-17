import configparser
import glob
import json
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import io
from PIL import Image
import json
from pydub import AudioSegment

def load_video_files(vid, params):
    return sorted(glob.glob(os.path.join(params['base_directory'], params['original_frames'], f"{vid}.mp4")))

def load_audio_files(vid, params):
    return sorted(glob.glob(os.path.join(params['base_directory'], params['original_frames'], f"{vid}.m4a")))

def load_key_video_files(vid, params):
    return sorted(glob.glob(os.path.join(params['base_directory'], params['keyframes'], f"{vid}.mp4")))

def load_embedding_files(vid, params):
    return sorted(glob.glob(os.path.join(params['base_directory'], params['originalembeddings'], f"{vid}.npy")))

def load_keyframe_embedding_files(vid, params):
    return sorted(glob.glob(os.path.join(params['base_directory'], params['embeddings'], f"{vid}.npy")))

def process_files(sample):
    result = {}
    for key, value in sample.items():
        if key.endswith("json"):
            result[key] = json.loads(value)
        elif key.endswith("npy"):
            result[key] = np.load(io.BytesIO(value))
        elif key.endswith(".png"):
            result[key] = Image.open(io.BytesIO(value)).convert("RGB")
        elif key.endswith("m4a") or key.endswith("flac") or key.endswith("mp3"):
            audio_format = "m4a" if key.endswith("m4a") else ("flac" if key.endswith("flac") else "mp3")
            audio_file = io.BytesIO(value)
            try:
                result[key] = AudioSegment.from_file(audio_file, format=audio_format)
            except Exception as e:
                print(f"Error processing {key}: {e}")
        elif key.endswith("txt"):
            try:
                text_content = value.decode('utf-8')
                result[key] = text_content
            except Exception as e:
                print(f"Error decoding {key}: {e}")
    return result

def load_embedding_values(embedding_files):
    if not embedding_files:
        raise ValueError("No embedding files provided.")
    loaded_arrays = [np.load(file) for file in embedding_files]
    if not any(len(arr) > 0 for arr in loaded_arrays):
        raise ValueError("Failed to load any arrays from embedding files.")
    return normalize(np.concatenate(loaded_arrays, axis=0), axis=1)

def get_video_duration(video_files):
    if not video_files:
        raise ValueError("No video files provided")
    vid_cap = cv2.VideoCapture(video_files)
    if not vid_cap.isOpened():
        vid_cap = cv2.VideoCapture(video_files[0])
        if not vid_cap.isOpened():
            raise IOError("Failed to open video file")
    total_duration = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(vid_cap.get(cv2.CAP_PROP_FPS))
    vid_cap.release()
    return total_duration

def get_all_video_ids(directory):
    return [int(os.path.basename(f).split('.')[0]) for f in glob.glob(f"{directory}/*.mp4")]