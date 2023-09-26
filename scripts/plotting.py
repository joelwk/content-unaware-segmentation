import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors
from segment_processing import calculate_successor_distance

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
