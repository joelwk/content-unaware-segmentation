import numpy as np
import configparser
import glob
import os
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
from imagehash import phash
from typing import List, Tuple, Union, Optional,Dict
from pipeline import delete_associated_files, parse_args, generate_config
import load_data as ld
from pipeline import read_config

def read_thresholds_config(section: str = 'thresholds') -> dict:
    params = read_config(section=section)
    return {key: None if params.get(key) in [None, 'None'] else float(params.get(key)) 
        for key in ['successor_value', 'phash_threshold','max_duration']}

def check_for_new_segment(distances: Union[np.ndarray, List[float]], 
                          successor_distances: Union[np.ndarray, List[float]], 
                          thresholds: Dict[str, Optional[float]]) -> List[int]:
    new_segments = []
    num_frames = len(distances) if distances.ndim == 1 else distances.shape[0]
    for i in range(num_frames - 1):
        avg_distance_frame_i = np.mean(distances[i])
        std_dev_frame_i = np.std(distances[i])
        threshold_frame_i = avg_distance_frame_i + 0.5 * std_dev_frame_i
        successor_distance = successor_distances[i]
        comparison_value = thresholds['successor_value'] or threshold_frame_i
        if float(successor_distance) > float(comparison_value) and i < num_frames - 1:
            new_segments.append(i)
    return new_segments
    
def calculate_successor_distance(embeddings: List[np.ndarray]) -> List[float]:
    '''Normalized successor distance between embeddings.'''
    if len(embeddings) < 2:
        print("Insufficient number of embeddings.")
        return []
    emb_current = np.array(embeddings[:-1])
    emb_next = np.array(embeddings[1:])
    euclidean_distance = np.linalg.norm(emb_next - emb_current, axis=1)
    min_val = np.min(euclidean_distance)
    max_val = np.max(euclidean_distance)
    normalized_distances = np.zeros_like(euclidean_distance) if max_val == min_val else \
        (euclidean_distance - min_val) / (max_val - min_val)
    return normalized_distances

def filter_keyframes_based_on_phash(frames: List[np.ndarray], keyframe_timestamps: List[float], thresholds: Dict[str, Optional[float]]) -> List[float]:
    phash_threshold = thresholds['phash_threshold']
    filtered_timestamps = []
    for idx1, frame1 in enumerate(frames):
        for idx2, frame2 in enumerate(frames[idx1+1:], start=idx1+1):
            hamming_distance = calculate_video_frame_phash_similarity(frame1, frame2)
            phash_threshold = thresholds['phash_threshold']
            if phash_threshold is not None and float(hamming_distance) > float(phash_threshold):
                filtered_timestamps.append(keyframe_timestamps[idx1])
    return filtered_timestamps

def calculate_video_frame_phash_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    hash1 = phash(img1)
    hash2 = phash(img2)
    hamming_distance = hash1 - hash2
    return hamming_distance

def get_segmented_and_filtered_frames(video_files: List[str], keyframe_files: List[str],embedding_values: List[np.ndarray],thresholds: Dict[str, Optional[float]]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    args = parse_args()
    config = {
        "local": generate_config("./datasets"),}
    selected_config = config[args.mode]
    frame_embedding_pairs = []
    timestamps = []
    video_id = None
    try:
        total_duration = ld.get_video_duration(video_files)
        for vid_path in keyframe_files:
            video_id = int(os.path.basename(vid_path).split('.')[0])
            vid_cap = cv2.VideoCapture(vid_path)
            for emb_idx, embedding in enumerate(embedding_values):
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, emb_idx)
                success, frame = vid_cap.read()
                if not success:
                    break
                timestamp = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                frame_embedding_pairs.append((frame, embedding))
                timestamps.append(timestamp)
            vid_cap.release()
        if thresholds['phash_threshold'] is not None:
            frames = [frame for frame, _ in frame_embedding_pairs]
            filtered_timestamps = filter_keyframes_based_on_phash(frames, timestamps, thresholds)
            frame_embedding_pairs = [(frame, emb) for frame, emb, ts in zip(frames, embedding_values, timestamps) if ts in filtered_timestamps]
        if len(frame_embedding_pairs) == 0:
            delete_associated_files(video_id, selected_config)
            print(f"No keyframes remaining after filtering for video ID {video_id}. Associated files deleted.")
            return [], []
        if len(frame_embedding_pairs) != len(timestamps):
            delete_associated_files(video_id, selected_config)
            print("Mismatch in the number of frames and timestamps after filtering.")
        timestamps = [ts for ts in timestamps if ts in filtered_timestamps]
        return frame_embedding_pairs, timestamps
    except Exception as e:
        if video_id is not None:
            delete_associated_files(video_id, selected_config)
            print(f"An error occurred during processing: {e}. Associated files for video ID {video_id} have been deleted.")
        else:
            print(f"An error occurred: {e}, but no video ID was available to delete associated files.")
        return [], []