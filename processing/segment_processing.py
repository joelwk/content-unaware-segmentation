import numpy as np
import configparser
import glob
import os
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from load_data import read_config

def check_for_new_segment(distances, successor_distances):
    params = read_config(section='thresholds')
    successor_value = params.get('successor_value', None)
    new_segments = []
    
    if distances.ndim == 1:
        num_frames = len(distances)
    elif distances.ndim == 2:
        num_frames, _ = distances.shape
    
    if successor_value is not None:
        successor_value = float(successor_value)
        
    for i in range(num_frames - 1):
        if distances.ndim == 1:
            avg_distance_frame_i = distances[i]
            std_dev_frame_i = np.std(distances)
        elif distances.ndim == 2:
            avg_distance_frame_i = np.mean(distances[i, :])
            std_dev_frame_i = np.std(distances[i, :])
        
        threshold_frame_i = avg_distance_frame_i + 0.5 * std_dev_frame_i
        successor_distance = successor_distances[i]
        comparison_value = successor_value if successor_value is not None else threshold_frame_i
        
        if float(successor_distance) > comparison_value:
            new_segments.append(i)
            
    return new_segments
    
def calculate_successor_distance(embeddings):
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

def calculate_distances_to_centroids(distances, indices):
    valid_indices = indices[indices < len(distances)]
    centroids = np.mean(distances[valid_indices], axis=1) if valid_indices.ndim > 1 else \
        np.mean(distances[valid_indices])
    return np.linalg.norm(distances[:, np.newaxis] - centroids, axis=1)