import glob
import json
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import configparser
from segment_processing import *
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from load_data import *
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from PIL import Image
from imagehash import phash

class SegmentSuccessorAnalyzer:
    def __init__(self, total_duration: float, embedding_values: np.ndarray, thresholds: Dict[str, Optional[float]],
                 max_segment_duration: Optional[int] = None) -> None:
        # Validate types
        if not isinstance(total_duration, float):
            raise TypeError("total_duration must be a float.")
        if not isinstance(embedding_values, np.ndarray):
            raise TypeError("embedding_values must be a numpy array.")
        
        # Read and store thresholds from config
        self.thresholds = self.read_thresholds_config()
        
        self.embedding_values = embedding_values
        self.total_duration = total_duration
        self.max_segment_duration = int(self.thresholds['max_duration']) if max_segment_duration is not None else None

    @staticmethod
    def read_thresholds_config(section: str = 'thresholds') -> dict:
        params = read_config(section=section)
        return {key: None if params.get(key) in [None, 'None'] else float(params.get(key)) 
                for key in ['successor_value', 'phash_threshold']}

    def run(self, video_files: List[str], thresholds: Dict[str, Optional[float]], keyframe_files: List[str], save_dir: str) -> Tuple[List[np.ndarray], List[float]]:
        # Use stored thresholds if none are provided
        thresholds = thresholds or self.thresholds
        frame_embedding_pairs, timestamps = get_segmented_and_filtered_frames(video_files, keyframe_files,self.embedding_values, thresholds)
        # Check for edge cases
        if len(frame_embedding_pairs) < 2:
            print(f"Insufficient number of frame embeddings for {video_files}. Skipping analysis.")
            return [], []
        try:
            temporal_embeddings = np.array([emb for _, emb in frame_embedding_pairs])
            distances = np.linalg.norm(temporal_embeddings[1:] - temporal_embeddings[:-1], axis=1)
        except AxisError as e:
            print(f"An AxisError occurred while processing {video_files}: {e}. Skipping analysis.")
            return [], []
        successor_distance = calculate_successor_distance(self.embedding_values)
        initial_new_segments = check_for_new_segment(distances, successor_distance, thresholds)
        new_segments = self.calculate_new_segments(initial_new_segments, timestamps)
        self.save_keyframes(frame_embedding_pairs, new_segments, distances, successor_distance, timestamps, save_dir)

    def calculate_new_segments(self, initial_new_segments, timestamps):
        if self.max_segment_duration is None:
            return initial_new_segments
        new_segments = []
        last_timestamp = timestamps[initial_new_segments[0]]
        new_segments.append(initial_new_segments[0])
        for i in range(1, len(initial_new_segments)):
            current_timestamp = timestamps[initial_new_segments[i]]
            if (current_timestamp - last_timestamp) > self.max_segment_duration:
                acceptable_frame = self.find_acceptable_frame(initial_new_segments[i-1:i+1], timestamps, last_timestamp)
                new_segments.append(acceptable_frame)
                last_timestamp = timestamps[acceptable_frame]
            new_segments.append(initial_new_segments[i])
        return new_segments

    def find_acceptable_frame(self, intervening_frames, timestamps, last_timestamp):
        for j in range(intervening_frames[0], intervening_frames[1]):
            if (timestamps[j] - last_timestamp) <= self.max_segment_duration:
                return j
        return intervening_frames[0]

    def save_keyframes(self, frame_embedding_pairs, new_segments, distances, successor_distance, timestamps, save_dir):
        if not new_segments:
            print("No new segments found. Exiting save_keyframes.")
            return
        saved_keyframes = set()
        for segment_idx in new_segments:
            if segment_idx not in saved_keyframes:
                frame, _ = frame_embedding_pairs[segment_idx]
                save_path = os.path.join(save_dir, f'keyframe_{segment_idx}.png')
                cv2.imwrite(save_path, frame)
                saved_keyframes.add(segment_idx)

        # Plot key frames widist dynamic sizing and save them
        num_keyframes = len(new_segments)
        num_cols = 4
        num_rows = int(np.ceil(num_keyframes / num_cols))
        if num_rows <= 0 or num_cols <= 0:
            print("Invalid grid dimensions. Skipping grid plotting.")
            return
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        if num_keyframes == 1:
            axes = np.array([[axes]])
        flat_axes = axes.flatten()
        keyframe_data = {}
        for i, ax in enumerate(flat_axes[:num_keyframes]):
            ax.imshow(cv2.cvtColor(frame_embedding_pairs[new_segments[i]][0], cv2.COLOR_BGR2RGB))
            annotate_plot(ax,
                        idx=new_segments[i],
                        successor_sim=successor_distance,
                        distances=distances,
                        global_frame_start_idx=0,
                        window_idx=i,
                        segment_label=f"Fame{i}",
                        timestamp=timestamps[new_segments[i]]
                        )
            # Store the index and time frame for this keyframe
            keyframe_data[i] = {
                'index': new_segments[i],
                'time_frame': timestamps[new_segments[i]]
            }
        for i in range(num_keyframes, len(flat_axes)):
            flat_axes[i].axis('off')
        plt_path = os.path.join(save_dir, 'keyframes_grid.png')
        plt.savefig(plt_path)
        # Save keyframe data
        json_path = os.path.join(save_dir, 'keyframe_data.json')
        with open(json_path, 'w') as f:
            json.dump(keyframe_data, f)

def annotate_plot(ax, idx, successor_sim, distances, global_frame_start_idx, window_idx, segment_label, timestamp=None):
    title_elements = [
        f"{segment_label}",
        f"Successor Value: {successor_sim[idx]:.2f}"
    ]
    if timestamp is not None:
        title_elements.append(f"Timestamp: {timestamp}")
    ax.set_title("\n".join(title_elements), fontsize=8, pad=6)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"FrameIndex {global_frame_start_idx + idx}", markersize=8)]
    ax.legend(handles=legend_elements, fontsize=6)

def run_analysis(analyzer_class, specific_videos=None):
    thresholds = read_thresholds_config()  
    params = read_config(section="directory")
    video_ids = get_all_video_ids(params['originalframes'])
    if specific_videos is not None:
        video_ids = [vid for vid in video_ids if vid in specific_videos]
    for video in video_ids:
      
        video_files = load_video_files(video, params)
        if not video_files:
            print(f"No video files found for video: {video}. Skipping analysis.")
            continue
        
        key_video_files = load_key_video_files(video, params)
        keyframe_embedding_files = load_keyframe_embedding_files(video, params)
        
        embedding_values = load_embedding_values(keyframe_embedding_files)
        total_duration = get_video_duration(video_files)
        save_dir = f"./output/keyframes/{video}"
        os.makedirs(save_dir, exist_ok=True)
        analyzer = analyzer_class(total_duration, embedding_values,thresholds)
        analyzer.run(video_files,thresholds, key_video_files, save_dir)
        
