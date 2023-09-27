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
from plotting import annotate_plot
from load_data import *

class SegmentSuccessorAnalyzer:
    # Initialize the class with video duration and embedding values
    def __init__(self, total_duration, embedding_values):
        self.embedding_values = embedding_values
        self.total_duration = total_duration

    # Function to get the timestamps for each frame and embedding combination - this creates the frame-embedding pairs
    def get_segmented_frames_and_embeddings(self, video_files):
        frame_embedding_pairs = []
        timestamps = []
        for vid_path in video_files:
            vid_cap = cv2.VideoCapture(vid_path)
            for emb_idx, embedding in enumerate(self.embedding_values):
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, emb_idx)
                success, frame = vid_cap.read()
                if not success:
                    break
                timestamp = (self.total_duration / len(self.embedding_values)) * emb_idx
                frame_embedding_pairs.append((frame, embedding))
                timestamps.append(timestamp)
            vid_cap.release()
        return frame_embedding_pairs, timestamps

    def run(self, video_files, save_dir):
        keyframe_embeddings = []
        keyframe_timestamps = []
        frame_embedding_pairs, timestamps = self.get_segmented_frames_and_embeddings(video_files)
        temporal_embeddings = np.array([emb for _, emb in frame_embedding_pairs])
        distances = np.linalg.norm(temporal_embeddings[1:] - temporal_embeddings[:-1], axis=1)
        # Call to an external function to calculate successor distances
        successor_distance = calculate_successor_distance(temporal_embeddings)
        # Call to an external function to identify new segments
        new_segments = check_for_new_segment(distances, successor_distance)

        # Plot key frames with dynamic sizing and save them
        # Calculate rows and columns for subplots
        num_keyframes = len(new_segments)
        num_cols = 4
        num_rows = int(np.ceil(num_keyframes / num_cols))
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
        save_single_keyframes(frame_embedding_pairs, new_segments, distances, successor_distance, timestamps, save_dir)
        return keyframe_embeddings, keyframe_timestamps

# Function to annotate the plot
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
    params = read_config(section="directory")
    video_ids = get_all_video_ids(params['originalframes'])
    if specific_videos is not None:
        video_ids = [vid for vid in video_ids if vid in specific_videos]
    for video in video_ids:
        video_files = load_video_files(video, params)
        key_video_files = load_key_video_files(video, params)
        embedding_files = load_embedding_files(video, params)
        embedding_values = load_embedding_values(embedding_files)
        # Use full video duration
        total_duration = get_video_duration(video_files)

        save_dir = f"./output/keyframes/{video}"
        os.makedirs(save_dir, exist_ok=True)
        analyzer = analyzer_class(total_duration, embedding_values)
        # Then use video with key frames
        analyzer.run(key_video_files, save_dir)

def save_single_keyframes(frame_embedding_pairs, new_segments, distances, successor_distance, timestamps, save_dir):
    for i, idx in enumerate(new_segments):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cv2.cvtColor(frame_embedding_pairs[idx][0], cv2.COLOR_BGR2RGB))
        # Annotate the plot
        annotate_plot(ax,
                      idx=idx,
                      successor_sim=successor_distance,
                      distances=distances,
                      global_frame_start_idx=0,
                      window_idx=i,
                      segment_label="FrameIndex",
                      timestamp=timestamps[idx]
                     )
        # Save the individual image
        plt_path = os.path.join(save_dir, f'keyframe_{idx}_{timestamps[idx]:.2f}.png')
        plt.savefig(plt_path)
        plt.close(fig)