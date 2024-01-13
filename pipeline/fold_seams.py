import os
from imagehash import phash
import json
import shutil
import cv2
import numpy as np
from PIL import Image
import cv2
import subprocess
from typing import List, Tuple, Union, Optional
from pipeline import read_config
from load_data import get_all_video_ids, load_video_files, load_audio_files, load_key_video_files, load_embedding_files, load_embedding_values, get_video_duration, load_keyframe_embedding_files
from segmentation_processing import get_segmented_and_filtered_frames, filter_keyframes_based_on_phash, calculate_successor_distance, check_for_new_segment, read_thresholds_config,delete_associated_files

def segment_video_using_keyframes_and_embeddings(video_path, keyframe_clip_output_dir, keyframe_timestamps, thresholds, suffix_=None):
    # Validate types
    if not isinstance(video_path, str):
        raise TypeError("video_path must be a string.")
    if not isinstance(keyframe_clip_output_dir, str):
        raise TypeError("keyframe_clip_output_dir must be a string.")
    if not isinstance(keyframe_timestamps, list):
        raise TypeError("keyframe_timestamps must be a list.")
    directories = read_config(section="directory")
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

def main(segment_video, segment_audio, specific_videos):
    params = read_config(section="directory")
    base_directory = params['base_directory']
    thresholds = read_thresholds_config()  # Read thresholds for consistency
    # Determine which videos to process
    video_ids = get_all_video_ids(os.path.join(base_directory, params['original_frames'])) if specific_videos is None else specific_videos
    for vid in video_ids:
        audio_files, video_files, key_video_files, embedding_files, keyframe_data = setup_for_video_audio(vid, params)
        # Skip this video if setup_for_video_audio returned None (due to an exception)
        if any(v is None for v in [audio_files, video_files, key_video_files, embedding_files, keyframe_data]):
            continue
        keyframe_timestamps = [data['time_frame'] for data in keyframe_data.values()]
        if segment_video:
            clip_output = os.path.join(base_directory, params['output'], params['keyframe_clip_output'], str(vid))
            output_path = os.path.join(base_directory, params['keyframe_clip_embeddings_output'])
            os.makedirs(clip_output, exist_ok=True)
            segment_video_using_keyframes_and_embeddings(key_video_files[0], output_path, clip_output, keyframe_timestamps, thresholds)
        if segment_audio:
            os.makedirs(output_path, exist_ok=True)
            segment_audio_using_keyframes(audio_files[0], output_path, keyframe_timestamps, thresholds, suffix_='_fromaudio_filtered')

def setup_for_video_audio(vid, params):
    try:
        # Load files and data required for both video and audio segmentation
        base_directory = params['base_directory']
        video_files = load_video_files(str(vid), params)
        key_video_files = load_key_video_files(str(vid), params)
        embedding_files = load_keyframe_embedding_files(str(vid), params)
        embedding_values = load_embedding_values(embedding_files)
        audio_files = load_audio_files(str(vid), params)
        # Read keyframe data
        json_path = os.path.join(".", base_directory, params['output'], params['keyframes'], str(vid), "keyframe_data.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No keyframe_data.json found for video id {vid}.")
        with open(json_path, 'r') as f:
            keyframe_data = json.load(f)
        audio_clip_output = os.path.join(base_directory, params['output'], params['keyframe_audio_clip_output'], str(vid))
        os.makedirs(audio_clip_output, exist_ok=True)
        return audio_files, video_files, key_video_files, embedding_files, keyframe_data
    except FileNotFoundError as e:
        print(e)
        # Removing directory associated with the video
        video_dir = os.path.join(base_directory, params['keyframe_output'], str(vid))
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
            print(f"Removed directory {video_dir} due to missing keyframe data.")
            delete_associated_files(str(vid), params)
        return None, None, None, None, None  # Return five None values
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None, None 