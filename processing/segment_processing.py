import numpy as np
import configparser
import glob
import os
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from load_data import read_config
from PIL import Image
from imagehash import phash
from load_data import *
from typing import List, Tuple, Union, Optional,Dict

def read_thresholds_config(section: str = 'thresholds') -> dict:
    params = read_config(section=section)
    return {key: None if params.get(key) in [None, 'None'] else float(params.get(key)) 
            for key in ['successor_value', 'phash_threshold']}

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
        
        if float(successor_distance) > float(comparison_value):
            print(f"New segment detected at frame {i + 1}")  ## Marks a new segment starting at next frame.

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

def calculate_distances_to_centroids(distances: np.ndarray, 
                                     indices: np.ndarray) -> np.ndarray:
    valid_indices = indices[indices < len(distances)]
    centroids = np.mean(distances[valid_indices], axis=1) if valid_indices.ndim > 1 else \
        np.mean(distances[valid_indices])
    return np.linalg.norm(distances[:, np.newaxis] - centroids, axis=1)

def filter_keyframes_based_on_phash(frames: List[np.ndarray], 
                                    keyframe_timestamps: List[float], 
                                    thresholds: Dict[str, Optional[float]]) -> List[float]:
    phash_threshold = thresholds['phash_threshold']
    filtered_timestamps = []
    for idx1, frame1 in enumerate(frames):
        for idx2, frame2 in enumerate(frames[idx1+1:], start=idx1+1):
            hamming_distance = calculate_video_frame_phash_similarity(frame1, frame2)
            phash_threshold = thresholds['phash_threshold']
            if phash_threshold is not None and float(hamming_distance) > float(phash_threshold):
                filtered_timestamps.append(keyframe_timestamps[idx1])
    return filtered_timestamps

def calculate_video_frame_phash_similarity(frame1: np.ndarray, 
                                           frame2: np.ndarray) -> float:
    img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    hash1 = phash(img1)
    hash2 = phash(img2)
    hamming_distance = hash1 - hash2
    return hamming_distance

def get_segmented_and_filtered_frames(video_files: List[str], keyframe_files: List[str],
                                      embedding_values: List[np.ndarray], 
                                      thresholds: Dict[str, Optional[float]]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Extracts keyframes from the video and pairs them with corresponding embeddings, then applies perceptual hashing for filtering.
    Parameters:
        video_files (List[str]): List of paths to the original video files.
        keyframe_files (List[str]): List of paths to the video files containing keyframes.
        embedding_values (List[np.ndarray]): List of embeddings for each keyframe.
        thresholds (Dict[str, Optional[float]]): A dictionary containing threshold values, e.g., for perceptual hashing.
    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]: A list of tuples where each tuple contains a frame and its corresponding embedding, and a list of timestamps for each keyframe.
    """
    frame_embedding_pairs = []
    timestamps = []
    total_duration = get_video_duration(video_files)
    for vid_path in keyframe_files:
        vid_cap = cv2.VideoCapture(vid_path)
        for emb_idx, embedding in enumerate(embedding_values):
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, emb_idx)
            success, frame = vid_cap.read()
            if not success:
                break
            timestamp = (total_duration / len(embedding_values)) * emb_idx
            frame_embedding_pairs.append((frame, embedding))
            timestamps.append(timestamp)
        vid_cap.release()
    if thresholds['phash_threshold'] is not None:
        frames = [frame for frame, _ in frame_embedding_pairs]
        filtered_timestamps = filter_keyframes_based_on_phash(frames, timestamps, thresholds)
        frame_embedding_pairs = [(frame, emb) for frame, emb, ts in zip(frames, embedding_values, timestamps) if ts in filtered_timestamps]
        timestamps = [ts for ts in timestamps if ts in filtered_timestamps]
    return frame_embedding_pairs, timestamps



def get_segmented_frames_and_embeddings(video_files: List[str], 
                                        embedding_values: List[np.ndarray], 
                                        total_duration: float, 
                                        start_idx: int, 
                                        end_idx: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], np.ndarray]:
    """
    Extracts the subset of frames and embeddings for a given window.
    Parameters:
        video_files (List[str]): List of paths to video files.
        embedding_values (List[np.ndarray]): List of embeddings.
        total_duration (float): Total duration of the video.
        start_idx (int): Starting index for the segment.
        end_idx (int): Ending index for the segment.
    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray, float]], np.ndarray]: List of tuples containing frame, embedding, and timestamp, and an array of segmented embeddings.
    """
    frame_embedding_pairs = []
    segmented_embeddings = []
    
    for vid_path in video_files:
        vid_cap = cv2.VideoCapture(vid_path)
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(vid_cap.get(cv2.CAP_PROP_FPS))
        avg_duration_per_frame = float(round(total_duration / total_frames, 4))
        end_idx = end_idx - 1
        avg_duration_per_embedding = float(round(total_duration / end_idx, 4))
        print(f"Duration in seconds: {total_duration} seconds")
        print(f"Average Duration per frame: {avg_duration_per_frame} seconds")
        print(f"Average Duration per embedding: {avg_duration_per_embedding} seconds")
        print(f"____________________________________________________________________")
        # Validate start and end indices
        if start_idx >= total_frames or end_idx > total_frames:
            return None, None
        for emb_idx in range(start_idx, end_idx):
            if emb_idx >= len(embedding_values):
                break
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, emb_idx)
            success, frame = vid_cap.read()
            if not success:
                break
            timestamp = (total_duration / len(embedding_values)) * emb_idx
            frame_embedding_pairs.append((frame, embedding_values[emb_idx], timestamp))
            segmented_embeddings.append(embedding_values[emb_idx])
        vid_cap.release()
    return frame_embedding_pairs, np.array(segmented_embeddings)
