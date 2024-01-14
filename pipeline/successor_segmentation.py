import glob
import json
import shutil
import logging
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import configparser
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from PIL import Image
from imagehash import phash
from matplotlib.patches import Rectangle
from pipeline import parse_args, generate_config,delete_associated_files,read_config,string_to_bool,is_directory_empty
from segmentation_processing import get_segmented_and_filtered_frames, calculate_successor_distance, check_for_new_segment, read_thresholds_config
import load_data as ld 

class SegmentSuccessorAnalyzer:
    def __init__(self, embedding_values: np.ndarray, max_segment_duration: Optional[int] = None) -> None:
        self.thresholds = read_thresholds_config(section="thresholds")
        if not isinstance(embedding_values, np.ndarray):
            raise TypeError("embedding_values must be a numpy array.")
        self.embedding_values = embedding_values
        self.max_segment_duration = int(self.thresholds['max_duration']) if max_segment_duration is None else max_segment_duration
        
    def run(self, video_files: List[str], thresholds: Dict[str, Optional[float]], keyframe_files: List[str], save_dir: str) -> Tuple[List[np.ndarray], List[float]]:
        directories = read_config(section="directory")
        config_params = read_config(section="config_params")
        frame_embedding_pairs, timestamps = get_segmented_and_filtered_frames(video_files, keyframe_files, self.embedding_values, thresholds)
        if len(frame_embedding_pairs) < 2:
            video_id = int(os.path.basename(video_files[0]).split('.')[0])
            delete_associated_files(video_id, directories)
            raise ValueError(f"No frame embeddings found for video ID {video_id}. Associated files deleted.")
        try:
            temporal_embeddings = np.array([emb for _, emb in frame_embedding_pairs])
            distances = np.linalg.norm(temporal_embeddings[1:] - temporal_embeddings[:-1], axis=1)
        except AxisError as e:
            video_id = int(os.path.basename(video_files[0]).split('.')[0])
            delete_associated_files(video_id, directories)
            raise ValueError(f"An AxisError occurred while processing video ID {video_id}: {e}. Associated files deleted.")
        successor_distance = calculate_successor_distance(self.embedding_values)
        initial_new_segments = check_for_new_segment(distances, successor_distance, thresholds)
        new_segments = self.calculate_new_segments(initial_new_segments, timestamps)
        self.save_keyframes(frame_embedding_pairs, new_segments, distances, successor_distance, timestamps, save_dir, plot_grid=string_to_bool(config_params.get("plot_grid", "False")))

    def calculate_new_segments(self, initial_new_segments, timestamps):
        try:
            if self.max_segment_duration is None:
                return initial_new_segments
            new_segments = [initial_new_segments[0]]
            last_timestamp = timestamps[new_segments[0]]
            for i in range(0, len(initial_new_segments)):
                if i == 0:
                    new_segments.append(initial_new_segments[i])
                    last_timestamp = timestamps[initial_new_segments[i]]
                    continue
                if i >= len(timestamps):
                    print(f"Index {i} out of bounds for timestamps list. Skipping.")
                    continue
                current_timestamp = timestamps[initial_new_segments[i]]
                if (current_timestamp - last_timestamp) > self.max_segment_duration:
                    if i-1 < 0 or i+1 > len(timestamps):
                        print(f"Index out of bounds during segment calculation at indices {i-1} and {i+1}. Skipping this segment.")
                        continue
                    acceptable_frame = self.find_acceptable_frame(initial_new_segments[i-1:i+1], timestamps, last_timestamp)
                    if acceptable_frame < 0 or acceptable_frame >= len(timestamps):
                        print(f"Acceptable frame index {acceptable_frame} out of bounds. Skipping this segment.")
                        continue
                    if acceptable_frame not in new_segments:
                        new_segments.append(acceptable_frame)
                        last_timestamp = timestamps[acceptable_frame]
                if initial_new_segments[i] not in new_segments:
                    new_segments.append(initial_new_segments[i])
                    last_timestamp = timestamps[initial_new_segments[i]]
            return new_segments
        except IndexError as e:
            print(f"An index error occurred: {e}.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
            return []

    def find_acceptable_frame(self, intervening_frames, timestamps, last_timestamp):
        if intervening_frames[1] > len(timestamps):
            print(f"Index out of bounds: {intervening_frames[1]} for timestamps of length {len(timestamps)}")
        for j in range(intervening_frames[0], min(intervening_frames[1], len(timestamps))):
            if (timestamps[j] - last_timestamp) <= self.max_segment_duration:
                return j
        return intervening_frames[0]

    def save_keyframes(self, frame_embedding_pairs, new_segments, distances, successor_distance, timestamps, save_dir, plot_grid=True):
        if len(frame_embedding_pairs) != len(timestamps):
            print("Mismatch in the number of frames and timestamps. Exiting save_keyframes.")
            return
        keyframe_data = {}
        segment_counter = 0
        if plot_grid:
            num_frames = len(frame_embedding_pairs)
            num_cols = 4
            num_rows = int(np.ceil(num_frames / num_cols))
            if num_rows <= 0 or num_cols <= 0:
                print("Invalid grid dimensions. Skipping grid plotting.")
                return
            max_fig_width, max_fig_height = 16000, 16000 
            fig_width, fig_height = 4 * num_cols, 4 * num_rows
            if fig_width > max_fig_width or fig_height > max_fig_height:
                scale_factor = min(max_fig_width / fig_width, max_fig_height / fig_height)
                fig_width, fig_height = scale_factor * fig_width, scale_factor * fig_height
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
            if num_frames == 1:
                axes = np.array([[axes]])
            flat_axes = axes.flatten()
            for i, ax in enumerate(flat_axes[:num_frames]):
                frame, _ = frame_embedding_pairs[i]
                if is_clear_image(frame):
                    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if i in new_segments:
                        segment_counter += 1
                    annotate_plot(ax, idx=i, successor_sim=successor_distance, distances=distances,
                                  global_frame_start_idx=0, window_idx=i,
                                  segment_label=f"Segment {segment_counter}", timestamp=timestamps[i])
            for ax in flat_axes[num_frames:]:
                ax.axis('off')
            plt.tight_layout()
            plt_path = os.path.join(save_dir, 'keyframes_grid.png')
            plt.savefig(plt_path)
        segment_counter = 0
        for i, (frame, _) in enumerate(frame_embedding_pairs):
            if is_clear_image(frame):
                if i in new_segments:
                    segment_counter += 1
                individual_keyframe_filename = f'keyframe_{i}_timestamp_{timestamps[i]:.2f}.png'
                individual_keyframe_path = os.path.join(save_dir, individual_keyframe_filename)
                cv2.imwrite(individual_keyframe_path, frame)
                keyframe_data[i] = {
                    'index': i,
                    'time_frame': timestamps[i],
                    'filename': individual_keyframe_filename}
        json_path = os.path.join(save_dir, 'keyframe_data.json')
        with open(json_path, 'w') as f:
            json.dump(keyframe_data, f)

def remove_whitespace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    return frame[y:y+h, x:x+w]

def annotate_plot(ax, idx, successor_sim, distances, global_frame_start_idx, window_idx, segment_label, timestamp=None):
    title_elements = [
        f"{segment_label}",
        f"Successor Value: {successor_sim[idx]:.2f}" if idx < len(successor_sim) else "No Successor Value"]
    if timestamp is not None:
        title_elements.append(f"Timestamp: {timestamp}")
    if idx >= len(successor_sim):
        print(f"Index out of bounds in annotate_plot: {idx} for successor_sim of length {len(successor_sim)}")
    ax.set_title("\n".join(title_elements), fontsize=8, pad=6)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Frame Index {global_frame_start_idx + idx}", markersize=8)]
    ax.legend(handles=legend_elements, fontsize=6)
    
def is_clear_image(frame, lower_bound=10, upper_bound=245):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_frame)
    return lower_bound < mean_intensity < upper_bound

def run_analysis(analyzer_class, specific_videos=None):
    thresholds = read_thresholds_config()  
    params = read_config(section="directory")
    video_ids = ld.get_all_video_ids(os.path.join(params['base_directory'], params['original_frames']))
    if specific_videos is not None:
        video_ids = [vid for vid in video_ids if vid in specific_videos]
    for video in video_ids:
        video = str(video)
        save_dir = os.path.join(params['base_directory'], params['keyframes'], video)
        try:
            keyframe_embedding_files = ld.load_keyframe_embedding_files(video, params)
            if not keyframe_embedding_files:
                raise ValueError(f"No embedding files provided for video {video}.")
            embedding_values = ld.load_embedding_values(keyframe_embedding_files)
            video_files = ld.load_video_files(video, params)
            if not video_files:
                raise ValueError(f"No video files found for video {video}.")
            key_video_files = ld.load_key_video_files(video, params)
            os.makedirs(save_dir, exist_ok=True)
            analyzer = analyzer_class(embedding_values, thresholds['max_duration'])
            analyzer.run(video_files, thresholds, key_video_files, save_dir)
            if is_directory_empty(save_dir):
                raise ValueError(f"No keyframes found after processing video {video}.")
        except ValueError as e:
            logging.warning(f"Error occurred for video {video}: {e}.")
            continue
    if not video_ids:
        print("All videos processed. Stopping script.")