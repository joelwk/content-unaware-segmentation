import os
from imagehash import phash
import json
import cv2
import numpy as np
from PIL import Image
import cv2
import subprocess
from typing import List, Tuple, Union, Optional
from load_data import read_config, get_all_video_ids, load_video_files, load_audio_files, load_key_video_files, load_embedding_files, load_embedding_values, get_video_duration, load_keyframe_embedding_files
from segment_processing import get_segmented_and_filtered_frames, filter_keyframes_based_on_phash, calculate_successor_distance, check_for_new_segment, calculate_distances_to_centroids, read_thresholds_config

def segment_video_using_keyframes_and_embeddings(video_path, keyframe_clip_output_dir, keyframe_timestamps, thresholds, suffix_=None):
    # Validate types
    if not isinstance(video_path, str):
        raise TypeError("video_path must be a string.")
    if not isinstance(keyframe_clip_output_dir, str):
        raise TypeError("keyframe_clip_output_dir must be a string.")
    if not isinstance(keyframe_timestamps, list):
        raise TypeError("keyframe_timestamps must be a list.")
    thresholds = read_config(section="thresholds")
    video_key_frames = cv2.VideoCapture(video_path)
    writer = None
    current_keyframe = 0
    frame_rate = int(video_key_frames.get(cv2.CAP_PROP_FPS))
    start_time = 0
    segment_idx = 0    
    while True:
        ret, frame = video_key_frames.read()
        if not ret:
            break
        current_time = video_key_frames.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_keyframe < len(keyframe_timestamps) - 1 and current_time >= keyframe_timestamps[current_keyframe]:
            end_time = current_time
            
            if "keyframe" in str(suffix_):
                # Calculate 10% tolerance and adjust start and end times
                tolerance = float(thresholds['tolerance'])  * (end_time - start_time)
                adjusted_start_time = start_time + tolerance
                adjusted_end_time = end_time - tolerance
            else:
                adjusted_start_time = start_time
                adjusted_end_time = end_time

            output_path = f"{keyframe_clip_output_dir}/keyframe_clip_{segment_idx}_{suffix_}.mp4"
            command = [
                'ffmpeg',
                '-ss', str(adjusted_start_time),
                '-to', str(adjusted_end_time),
                '-i', video_path,
                '-c:v', 'copy', '-c:a', 'copy',
                '-y', output_path
            ]
            subprocess.run(command)
            
            start_time = current_time
            segment_idx += 1
            current_keyframe += 1

    # Process the final segment
    if start_time < current_time:
        if "keyframe" in str(suffix_):
            # Calculate 10% tolerance and adjust start and end times for the final segment
            tolerance = float(thresholds['tolerance'])  * (current_time - start_time)
            adjusted_start_time = start_time + tolerance
            adjusted_end_time = current_time - tolerance
        else:
            adjusted_start_time = start_time
            adjusted_end_time = current_time
            
        output_path = f"{keyframe_clip_output_dir}/keyframe_clip_{segment_idx}_{suffix_}.mp4"
        command = [
            'ffmpeg',
            '-ss', str(adjusted_start_time),
            '-to', str(adjusted_end_time),
            '-i', video_path,
            '-c:v', 'copy', '-c:a', 'copy',
            '-y', output_path
        ]
        subprocess.run(command)
    video_key_frames.release()

def segment_audio_using_keyframes(audio_path, audio_clip_output_dir, keyframe_timestamps, thresholds, suffix_=None):
    start_time = 0
    segment_idx = 0
    current_time = 0.0  # Initialize current_time to 0
    for i, timestamp in enumerate(keyframe_timestamps):
        end_time = timestamp
        # Tolerance logic (similar to video)
        if "keyframe" in str(suffix_):
            tolerance = float(thresholds['tolerance'])  * (end_time - start_time)
            adjusted_start_time = start_time + tolerance
            adjusted_end_time = end_time - tolerance
        else:
            adjusted_start_time = start_time
            adjusted_end_time = end_time

        output_path = f"{audio_clip_output_dir}/keyframe_audio_clip_{segment_idx}_{suffix_}.m4a"
        command = [
            'ffmpeg',
            '-ss', str(adjusted_start_time),
            '-to', str(adjusted_end_time),
            '-i', audio_path,
            '-acodec', 'copy',
            '-y', output_path
        ]
        subprocess.run(command)

        start_time = end_time
        segment_idx += 1
        current_time = end_time

def main(specific_videos=None):
    params = read_config(section="directory")
    thresholds = read_thresholds_config()  # Read thresholds here for consistency
    # Validate types
    if specific_videos and not isinstance(specific_videos, list):
        raise TypeError("specific_videos must be a list or None.")
    
    video_ids = get_all_video_ids(params['originalframes']) if specific_videos is None else specific_videos
    for vid in video_ids:
        setup_for_video_audio(vid, params, thresholds)  # Pass thresholds as an argument

def setup_for_video_audio(vid, params, thresholds):
    # Validate types
    thresholds = read_thresholds_config()
    video_files = load_video_files(str(vid), params)
    key_video_files = load_key_video_files(str(vid), params)
    embedding_files = load_keyframe_embedding_files(str(vid), params)
    embedding_values = load_embedding_values(embedding_files)
    clip_output = f"./output/keyframe_clip/{vid}"
    os.makedirs(clip_output, exist_ok=True)
    json_path = os.path.join(".", "output", "keyframes", str(vid), "keyframe_data.json")
    if not os.path.exists(json_path):
        print(f"No keyframe_data.json found for video id {vid}. Skipping.")
        return
    with open(json_path, 'r') as f:
        keyframe_data = json.load(f)

    # Extract timestamps from the keyframe_data
    keyframe_timestamps = [data['time_frame'] for data in keyframe_data.values()]
    # Using the union of timestamps to segment video
    segment_video_using_keyframes_and_embeddings(key_video_files[0], clip_output, keyframe_timestamps,thresholds, suffix_='_fromkeyvideo_filtered')
    
    # New audio segmentation
    audio_files = load_audio_files(str(vid), params)
    audio_clip_output = f"./output/keyframe_audio_clip/{vid}"
    os.makedirs(audio_clip_output, exist_ok=True)
    segment_audio_using_keyframes(audio_files[0], audio_clip_output, keyframe_timestamps, thresholds, suffix_='_fromaudio_filtered')