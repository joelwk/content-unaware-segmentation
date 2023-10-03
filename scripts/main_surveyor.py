import glob
import os
import cv2
import pickle
import numpy as np
from sklearn.preprocessing import normalize
import configparser
from segment_processing import *
from load_data import *
from embedding_surveyor import SlidingWindowAnalyzer
from plotting import *

def run_analysis(specific_videos=None):
    directory = read_config(section="directory")
    thresholds = read_config(section="thresholds")
    video_ids = get_all_video_ids(directory['originalframes'])
    
    global_video_analysis = []
    if specific_videos is not None:
        video_ids = [vid for vid in video_ids if vid in specific_videos]
    for video in video_ids:
        video_files = load_video_files(video, directory)
        key_video_files = load_key_video_files(video, directory)
        embedding_files = load_embedding_files(video, directory)
        key_embedding_files = load_keyframe_embedding_files(video,directory)
        embedding_values = load_embedding_values(key_embedding_files)
        total_duration = get_video_duration(video_files)
        save_dir = f"./output/{video}"
        os.makedirs(save_dir, exist_ok=True)
        analyzer = SlidingWindowAnalyzer(total_duration, embedding_values, window_size=int(thresholds['window_size']), step_size=int(thresholds['step_size']))
        analyzer.initialize_thresholds(avg_distance=float(thresholds['avg_distance']), std_dev=float(thresholds['std_dev']))
        analyzer.run(key_video_files, total_duration, save_dir, global_video_analysis)
    # Save the global_video_analysis list to a pickle file
    with open(f"{directory['evaluations']}/global_video_analysis{video}.pkl", "wb") as f:
        pickle.dump(global_video_analysis, f)
    evaluate_embedding_statistics(video)
if __name__ == "__main__":
    run_analysis()