import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle
import numpy as np
from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors
from segment_processing import calculate_successor_distance
from load_data import *

def calculate_dynamic_perplexity(global_frame_start_idx, num_embeddings):
    return np.clip(global_frame_start_idx, 1, min(num_embeddings - 1, num_embeddings // 2))

def annotate_plot(ax, idx, successor_sim, distances, optimal_k, global_frame_start_idx, window_idx, segment_label, timestamp=None):
    title_elements = [
        f"{segment_label}",
        f"Window: {window_idx}",
        f"Successor Distance: {successor_sim[idx]:.2f}",
        f"Dist. to Centroid: {distances[idx]:.2f}",
        f"Optimal K: {optimal_k}"
    ]
    if timestamp:
        title_elements.append(f"Timestamp: {timestamp}")
    ax.set_title("\n".join(title_elements), fontsize=8, pad=6)
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w', label=f"Global Frame Idx {global_frame_start_idx + idx}", markersize=8)], fontsize=6)

def plot_frames(axes, frame_embedding_pairs, scores, distances, optimal_k, global_frame_start_idx, window_idx, new_segments):
    for idx, (frame, _, timestamp) in enumerate(frame_embedding_pairs):
        ax = axes[idx]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        if idx < len(scores):
            window_frame_label = f"Window Frame {idx+1} {'(New)' if new_segments and idx in new_segments else ''}"
            annotate_plot(ax, idx, scores, distances, optimal_k, global_frame_start_idx, window_idx, window_frame_label, timestamp)
    for ax in axes[len(frame_embedding_pairs):]:
        ax.axis('off')

def calculate_centroid_labels(reduced_embeddings, knn):
    _, indices = knn.kneighbors(reduced_embeddings)
    centroid_to_label = {}
    for i, knn_cluster in enumerate(reduced_embeddings[indices]):
        centroid = tuple(np.mean(knn_cluster, axis=0))
        centroid_to_label.setdefault(centroid, len(centroid_to_label))
    return centroid_to_label

def plot_and_save_tsne(reduced_embeddings, centroid_to_label, window_idx, save_dir, optimal_k):
    fig, ax = plt.subplots()
    for i, (x, y) in enumerate(reduced_embeddings):
        ax.scatter(x, y, color='blue')
        ax.annotate(f"E{i}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    for centroid, label in centroid_to_label.items():
        ax.scatter(*centroid, color='red')
        ax.annotate(f"C{label}", centroid, textcoords="offset points", xytext=(0,10), ha='center')
    plt.savefig(os.path.join(save_dir, f"tsne_scatter_{window_idx}.png"))

def combine_and_save_plots(wind_save_path, window_idx, save_dir):
    tsne_save_path = os.path.join(save_dir, f"tsne_scatter_{window_idx}.png")
    if not all(map(os.path.exists, [wind_save_path, tsne_save_path])):
        print(f"Files do not exist: {wind_save_path}, {tsne_save_path}")
        return
    images = [Image.open(x) for x in [wind_save_path, tsne_save_path]]
    widths, heights = zip(*(i.size for i in images))
    new_img = Image.new('RGB', (sum(widths), max(heights)))
    new_img.paste(images[0], (0, 0))
    new_img.paste(images[1], (widths[0], 0))
    new_img.save(os.path.join(save_dir, f"combined_{window_idx}.png"))
    plt.show()

def evaluate_embedding_statistics(ids):
    if isinstance(ids, int):  # Convert integer to single-item list
        ids = [ids]
    directory = read_config(section="directory")
    fig, axs = plt.subplots(5, 1, figsize=(15, 10), dpi=100) 
    if ids is None:
        ids = [filename.split("global_video_analysis")[1].split(".pkl")[0] for filename in glob.glob(f"{directory['evaluations']}global_video_analysis*.pkl")]
    for id in ids:
        with open(f"{directory['evaluations']}global_video_analysis{id}.pkl", 'rb') as f:
            data = pickle.load(f)
        all_distances_means = []
        successor_distances_means = []
        new_segments_count = []
        avg_window_len = []
        segment_density = []
        total_duration = data[0]['total_duration']
        x_axis_range = np.linspace(0, total_duration, len(data))
        for entry in data:
            all_distances_means.append(np.mean(entry['distances']))
            successor_distances_means.append(np.mean(entry['successor_distance']))
            new_segments_count.append(len(entry['new_segments']))
            avg_len = (entry['end_idx'] - entry['start_idx']) / len(entry['new_segments']) if entry['new_segments'] else 0
            avg_window_len.append(avg_len)
            segment_density.append(len(entry['new_segments']) / avg_len if avg_len else 0)
        # Plotting with increased line width and markers
        axs[0].plot(x_axis_range, all_distances_means, label=f'Mean All Distances (ID: {id})', linewidth=2, marker='o')
        axs[1].plot(x_axis_range, successor_distances_means, label=f'Mean Successor Distances (ID: {id})', linewidth=2, marker='x')
        axs[2].plot(x_axis_range, new_segments_count, label=f'New Segments Count (ID: {id})', linewidth=2, marker='s')
        axs[3].plot(x_axis_range, avg_window_len, label=f'Average Window Length (ID: {id})', linewidth=2, marker='d')
        axs[4].plot(x_axis_range, segment_density, label=f'Segment Density (ID: {id})', linewidth=2, marker='*')
    for ax, ylabel in zip(axs, ['Mean All Distances', 'Mean Successor Distances', 'New Segments Count', 'Average Window Length (s)', 'Segment Density']):
        ax.set_xlabel('Total Duration (s)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)  # Added grid for better readability
    plt.tight_layout()
    plt.show()