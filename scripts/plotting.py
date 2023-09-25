import os
import math
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from segment_processing import calculate_successor_distance

def calculate_dynamic_perplexity(global_frame_start_idx, num_embeddings):
    return max(1, min(global_frame_start_idx, num_embeddings - 1, num_embeddings // 2))

def annotate_plot(ax, idx, successor_sim, distances, optimal_k, global_frame_start_idx, window_idx, segment_label, timestamp=None):
    title_elements = [
        f"{segment_label}",
        f"Window: {window_idx}",
        f"Successor Distance: {successor_sim[idx]:.2f}",
        f"Dist. to Centroid: {distances[idx]:.2f}",
        f"Optimal K: {optimal_k}"
    ]
    if timestamp is not None:
        title_elements.append(f"Timestamp: {timestamp}")
    ax.set_title("\n".join(title_elements), fontsize=8, pad=6)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Global Frame Idx {global_frame_start_idx + idx}", markersize=8)]
    ax.legend(handles=legend_elements, fontsize=6)

def plot_frames(axes, frame_embedding_pairs, scores, distances, optimal_k, global_frame_start_idx, window_idx, new_segments):
    for idx, (frame, _, timestamp) in enumerate(frame_embedding_pairs):
        ax = axes[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.axis('off')
        if idx < len(scores) and idx < len(distances):
            window_frame_label = f"Window Frame {idx+1}"
            if new_segments and idx in new_segments:
                window_frame_label += " (New)"
            annotate_plot(ax, idx, scores, distances, optimal_k, global_frame_start_idx, window_idx, window_frame_label, timestamp=timestamp)
            
        for idx in range(len(frame_embedding_pairs), len(axes)):
            axes[idx].axis('off')

def calculate_centroid_labels(reduced_embeddings, knn):
    _, indices = knn.kneighbors(reduced_embeddings)
    centroid_to_label = {}
    label_counter = 0  # Starting from 0
    for i, (x, y) in enumerate(reduced_embeddings):
        knn_cluster = reduced_embeddings[indices[i, :]]
        centroid = tuple(np.mean(knn_cluster, axis=0))
        if centroid not in centroid_to_label:
            centroid_to_label[centroid] = label_counter
            label_counter += 1
    return centroid_to_label

def plot_and_save_tsne(reduced_embeddings, centroid_to_label, window_idx, save_dir, optimal_k):
    fig, ax = plt.subplots()
    # Plot and label the data points (embeddings)
    for i, (x, y) in enumerate(reduced_embeddings):
        ax.scatter(x, y, color='blue')
        ax.annotate(f"E{i}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    centroid_coords = np.array(list(centroid_to_label.keys()))
    for centroid, label in centroid_to_label.items():
        ax.scatter(centroid[0], centroid[1], color='red')
        ax.annotate(f"C{label}", centroid, textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_title('t-SNE Scatter Plot with Centroids and Embedding Labels')
    tsne_save_path = os.path.join(save_dir, f"tsne_scatter_{window_idx}.png")
    plt.savefig(tsne_save_path)

def combine_and_save_plots(wind_save_path, window_idx, save_dir):
    tsne_save_path = os.path.join(save_dir, f"tsne_scatter_{window_idx}.png")
    if os.path.exists(wind_save_path) and os.path.exists(tsne_save_path):
        images = [Image.open(x) for x in [wind_save_path, tsne_save_path]]
    else:
        print(f"Files do not exist: {wind_save_path}, {tsne_save_path}")
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    combined_save_path = os.path.join(save_dir, f"combined_{window_idx}.png")
    new_img.save(combined_save_path)
    plt.show()