import glob
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import configparser
from segment_processing import *
from load_data import *
from embedding_surveyor import SlidingWindowAnalyzer

def run_analysis(specific_videos=None):
    directory = read_config(section="directory")
    video_ids = get_all_video_ids(directory['originalframes'])

    # Filter based on specific videos if provided
    if specific_videos is not None:
        video_ids = [vid for vid in video_ids if vid in specific_videos]
    for video in video_ids:
        video_files = load_video_files(video, directory)
        key_video_files = load_key_video_files(video, directory)
        embedding_files = load_embedding_files(video, directory)
        embedding_values = load_embedding_values(embedding_files)
        total_duration = get_video_duration(video_files)

        save_dir = f"datasets/{video}"
        os.makedirs(save_dir, exist_ok=True)
        
        analyzer = SlidingWindowAnalyzer(total_duration, embedding_values, window_size=30, step_size=15)
        analyzer.initialize_thresholds()
        analyzer.run(key_video_files, save_dir)

if __name__ == "__main__":
    run_analysis()