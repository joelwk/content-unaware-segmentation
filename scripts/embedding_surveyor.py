import cv2
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import math
from sklearn.manifold import TSNE
from segment_processing import calculate_successor_distance, check_for_new_segment, calculate_distances_to_centroids
from plotting import calculate_dynamic_perplexity,plot_frames,calculate_centroid_labels,plot_and_save_tsne,combine_and_save_plots,annotate_plot
from load_data import read_config
from typing import Any, Tuple, List, Optional, Union

class SlidingWindowAnalyzer:
    def __init__(self, total_duration: float, embedding_values: Union[List[np.ndarray], np.ndarray], 
                 window_size: Optional[float] = None, step_size: Optional[float] = None, 
                 avg_distance: Optional[float] = None, std_dev: Optional[float] = None) -> None:
        # Validate types
        if not isinstance(total_duration, float):
            raise TypeError("total_duration must be a float.")
        if not isinstance(embedding_values, (list, np.ndarray)):
            raise TypeError("embedding_values must be a list or numpy array.")
        
        self.thresholds = self.read_thresholds_config()
        # Initialize instance variables
        self.embedding_values = np.array(embedding_values)
        self._initialize_window_params(total_duration, window_size, step_size)
        self.segment_statistics = []
        self.knn = NearestNeighbors()
        self.total_duration = total_duration
        self.initialize_thresholds(avg_distance, std_dev)

    @staticmethod
    def read_thresholds_config(section: str = 'thresholds') -> dict:
        params = read_config(section=section)
        return {key: None if params.get(key) in [None, 'None'] else float(params.get(key)) 
                for key in ['successor_value', 'phash_threshold']}

    def _initialize_window_params(self, total_duration: float, window_size: Optional[float], step_size: Optional[float]) -> None:
        if not isinstance(total_duration, float):
            raise TypeError("total_duration must be a float.")
        # Calculate average time per embedding
        avg_sec_per_embedding = total_duration / len(self.embedding_values)
        # Initialize window and step parameters
        self.window_frame_count = math.ceil(window_size / avg_sec_per_embedding) if window_size else None
        self.step_frame_count = math.ceil(step_size / avg_sec_per_embedding) if step_size else None
        self.global_frame_start_idx = 0
        self.window_idx = 0

    def initialize_thresholds(self, avg_distance: Optional[float], std_dev: Optional[float]) -> None:
        if avg_distance and not isinstance(avg_distance, float):
            raise TypeError("avg_distance must be a float.")
        if std_dev and not isinstance(std_dev, float):
            raise TypeError("std_dev must be a float.")
        # Initialize distance thresholds for segmenting
        self.avg_distance_threshold = avg_distance
        self.std_dev_threshold = std_dev
        self.past_avg_distances = []
        self.past_std_devs = []

    def _window_indices(self):
        # Generate indices for window selection
        end_idx = self.window_frame_count
        while end_idx and end_idx <= len(self.embedding_values):
            yield self.global_frame_start_idx, end_idx
            self.global_frame_start_idx += self.step_frame_count
            end_idx += self.step_frame_count
            self.window_idx += 1

    def update_thresholds(self) -> None:
        # Update the thresholds based on recent history
        last_five_diffs_avg = np.mean(np.diff(self.past_avg_distances[-5:]))
        last_five_diffs_std = np.mean(np.diff(self.past_std_devs[-5:]))
        self.avg_distance_threshold += abs(last_five_diffs_avg) + 0.001
        self.std_dev_threshold += abs(last_five_diffs_std) + 0.001
        
    def check_for_new_segment(self, distances, successor_distances, thresholds=None):
        # Use stored thresholds if none are provided
        thresholds = thresholds or self.thresholds
        return check_for_new_segment(distances, successor_distances, thresholds)
    def calculate_successor_distance(self, embeddings):
        return calculate_successor_distance(embeddings)
    def calculate_distances_to_centroids(self, distances, indices):
        return calculate_distances_to_centroids(distances, indices)

    def calculate_optimal_k(self, embedding_scores, max_iter=1000):
        # Determine optimal K for KNN based on distance thresholds - adjust max_iter as needed
        # Uncomment print statements for debugging or to enable evaluation outputs
        if len(embedding_scores) == 0:
         #   print("The embedding_scores array is empty. Cannot proceed with KNN.")
            return None, None, None
        reshaped_scores = embedding_scores.reshape(-1, 1)
        self.knn = NearestNeighbors()
        self.knn.fit(reshaped_scores)
        K, iteration = 1, 0
        while iteration < max_iter:
            if K >= len(embedding_scores):
           #     print("Resetting K to 1 and updating thresholds.")
                self.update_thresholds()
                K = 1
            _, indices = self.knn.kneighbors(reshaped_scores, n_neighbors=K, return_distance=True)
            distances = self.calculate_distances_to_centroids(embedding_scores, indices)
            avg_distance, std_dev = np.mean(distances), np.std(distances)
           # print(f"Current K: {K}, Avg distance: {avg_distance}, Std dev: {std_dev}")
            self.past_avg_distances.append(avg_distance)
            self.past_std_devs.append(std_dev)
            if avg_distance < self.avg_distance_threshold and std_dev < self.std_dev_threshold:
              #  print(f"Optimal K found: {K}")
                return K, self.knn, distances
            K += 1
            iteration += 1
        print("Max iterations reached. Returning K=1 and thresholds.")
        return 1, self.knn, distances

    def plot_frames_and_tsne(self, frame_embedding_pairs, temporal_embeddings, window_idx, global_frame_start_idx, save_dir, new_segments=None):
        # Function for plotting frames and t-SNE embeddings
        if isinstance(temporal_embeddings, list):
          temporal_embeddings = np.array(temporal_embeddings)
        if temporal_embeddings.shape[0] < 2:
            print("Insufficient number of embeddings. Skipping t-SNE and KNN.")
            return
        # Initialize t-SNE with dynamic perplexity
        dynamic_perplexity = calculate_dynamic_perplexity(self.global_frame_start_idx, temporal_embeddings.shape[0])
        tsne = TSNE(perplexity=dynamic_perplexity)
        num_rows = math.ceil(math.sqrt(len(frame_embedding_pairs)))
        num_cols = math.ceil(len(frame_embedding_pairs) / num_rows)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10 * num_rows / num_cols))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        # KNN and t-SNE calculations
        distance = self.calculate_successor_distance(temporal_embeddings)
        optimal_k, _, distances = self.calculate_optimal_k(distance)
        if not isinstance(temporal_embeddings, np.ndarray):
            temporal_embeddings = np.array(temporal_embeddings)
        reduced_embeddings = tsne.fit_transform(temporal_embeddings)
        knn = NearestNeighbors(n_neighbors=optimal_k)
        knn.fit(reduced_embeddings)
        plot_frames(axes, frame_embedding_pairs, distance, distances, optimal_k, global_frame_start_idx, window_idx, new_segments)
        # Save and plot additional figures
        centroid_to_label = calculate_centroid_labels(reduced_embeddings, knn)
        # Save and plot figures
        wind_save_path = os.path.join(save_dir, f"window_segment_{window_idx}_{optimal_k}.png")
        plt.savefig(wind_save_path)
        plot_and_save_tsne(reduced_embeddings, centroid_to_label, window_idx, save_dir, optimal_k)
        combine_and_save_plots(wind_save_path, window_idx, save_dir)

    def get_segmented_frames_and_embeddings(self, video_files, start_idx, end_idx):
        '''Extracts the subset of frames and embeddings for a given window.'''
        frame_embedding_pairs = []
        segmented_embeddings = []
        for vid_path in video_files:
            vid_cap = cv2.VideoCapture(vid_path)
            frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
            if start_idx >= int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or end_idx > int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                print("Start or end index out of bounds.")
                return None, None
            for emb_idx in range(start_idx, end_idx):
                if emb_idx >= len(self.embedding_values):
                    break
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, emb_idx)
                success, frame = vid_cap.read()
                if not success:
                    break
                timestamp = (self.total_duration / len(self.embedding_values)) * emb_idx
                frame_embedding_pairs.append((frame, self.embedding_values[emb_idx], timestamp))
                segmented_embeddings.append(self.embedding_values[emb_idx])
            vid_cap.release()
        return frame_embedding_pairs, np.array(segmented_embeddings)

    def run(self, key_video_files, total_duration, save_dir, global_video_analysis):
        for start_idx, end_idx in self._window_indices():
            print(f"Processing window: start_idx={start_idx}, end_idx={end_idx}")
            is_last_window = end_idx - start_idx < self.window_frame_count and end_idx - start_idx > 0
            frame_embedding_pairs, temporal_embeddings = self.get_segmented_frames_and_embeddings(key_video_files, start_idx, end_idx)
            if frame_embedding_pairs is None or temporal_embeddings is None:
                print(f"Skipping window: start_idx={start_idx}, end_idx={end_idx}")
                continue
            _, _, distances = self.calculate_optimal_k(temporal_embeddings)
            successor_distance = self.calculate_successor_distance(temporal_embeddings)
            new_segments = self.check_for_new_segment(distances, successor_distance)
            global_video_analysis.append({
                'video_files': key_video_files,
                'total_duration': total_duration,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'distances': distances,
                'successor_distance': successor_distance,
                'new_segments': new_segments
            })
            if is_last_window:
                if len(temporal_embeddings) == 0 or len(new_segments) < 2:
                    print("The last window has insufficient data for KNN. Skipping this window.")
                    continue
                else:
                    print("The last window has sufficient data. Proceeding as usual.")
            self.plot_frames_and_tsne(frame_embedding_pairs, temporal_embeddings, self.window_idx, self.global_frame_start_idx, save_dir, new_segments=new_segments)
            