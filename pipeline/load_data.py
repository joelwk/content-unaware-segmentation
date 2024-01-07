import configparser
import glob
import json
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def load_video_files(vid, params):
    return sorted(glob.glob(f"{params['originalframes']}/{vid}.mp4"))

def load_audio_files(vid, params):
    return sorted(glob.glob(f"{params['originalframes']}/{vid}.m4a"))

def load_key_video_files(vid, params):
    return sorted(glob.glob(f"{params['keyframes']}/{vid}.mp4"))

def load_embedding_files(vid, params):
    return sorted(glob.glob(f"{params['originalembeddings']}/{vid}.npy"))

def load_keyframe_embedding_files(vid, params):
    return sorted(glob.glob(f"{params['embeddings']}/{vid}.npy"))
    
def load_embedding_values(embedding_files):
    if not embedding_files:
        raise ValueError("No embedding files provided.")
    loaded_arrays = [np.load(file) for file in embedding_files]
    if not any(len(arr) > 0 for arr in loaded_arrays):
        raise ValueError("Failed to load any arrays from embedding files.")
    return normalize(np.concatenate(loaded_arrays, axis=0), axis=1)
    
def get_video_duration(video_files):
    vid_cap = cv2.VideoCapture(video_files[0])
    total_duration = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(vid_cap.get(cv2.CAP_PROP_FPS))
    vid_cap.release()
    return total_duration

def get_all_video_ids(directory):
    return [int(os.path.basename(f).split('.')[0]) for f in glob.glob(f"{directory}/*.mp4")]