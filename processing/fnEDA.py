import cv2
import matplotlib.gridspec as gridspec
from load_data import *
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import cv2
import matplotlib.pyplot as plt
scatter_size = (9, 4)
def initialize_video(video_file):
    vid_cap = cv2.VideoCapture(video_file)
    frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
    return vid_cap, frame_rate

def visualize_frames(vid_cap, frame_indices, embedding_values):
    n_rows = int(np.ceil(len(frame_indices) / 5))
    fig = plt.figure(figsize=(15, 2 * n_rows + 5))
    gs = gridspec.GridSpec(n_rows + 1, 5, height_ratios=[1] * n_rows + [2])
    next_indices = frame_indices[1:] + [None]
    for i, emb_idx in enumerate(frame_indices):
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, emb_idx)
        success, frame = vid_cap.read()
        if not success:
            break
        next_idx = next_indices[i] if i < len(next_indices) else None
        similarity = round(euclidean_distances(embedding_values[emb_idx].reshape(1, -1),
                                               embedding_values[next_idx].reshape(1, -1))[0][0], 2) if next_idx is not None else 'NA'
        avg_embedding = round(float(np.mean(embedding_values[emb_idx])), 6)
        ax = plt.subplot(gs[i])
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Frame {emb_idx}\nAvg Embedding: {avg_embedding}\nSuccessor Similarity: {similarity}', fontsize=8)
    return fig, gs

def plot_embedding_and_similarity(fig, gs, frame_indices, embedding_values, is_average):
    ax1 = plt.subplot(gs[-5:])
    plot_indices = [idx for idx in frame_indices if idx < len(embedding_values)]
    next_indices = plot_indices[1:] + [None]
    
    ax1.bar(range(len(plot_indices)), 
            [euclidean_distances(embedding_values[i].reshape(1, -1), 
                                 embedding_values[next_indices[i]].reshape(1, -1))[0][0] if next_indices[i] is not None else 0 
             for i, _ in enumerate(plot_indices)], 
            color='green', 
            label='Successor Frame Similarity')
    ax1.set_xticks(range(len(plot_indices)))
    ax1.set_xticklabels(plot_indices)
    ax1.set_xlabel('Frame Embedding Index', fontsize=10)
    ax1.set_ylabel('Similarity', fontsize=10)
    ax2 = ax1.twinx()
    if is_average:
        plotted_values = [np.mean(embedding_values[i]) for i in plot_indices]
        ax2.set_ylabel('Avg. Embedding Vector', fontsize=10)
    else:
        # Example: Using the first dimension of the embedding for plotting
        plotted_values = [embedding_values[i][0] for i in plot_indices]
        ax2.set_ylabel('Specific Dimension Value', fontsize=10)
    ax2.scatter(range(len(plot_indices)), plotted_values, color='red', label='Embedding Vectors')
    for i, txt in enumerate(plotted_values):
        ax2.annotate(f"{txt:.6f}", (i, txt), textcoords="offset points", xytext=(0,10), ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))
    plt.title('Successor Similarity & Embedding Vectors Over Video Duration', fontsize=14)
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()


def view_pca_time(original_embeddings, key_embeddings, fps):
    # Convert frame indices to time in seconds
    original_time_seconds = np.arange(len(original_embeddings)) / fps
    key_time_seconds = np.arange(len(key_embeddings)) / fps

    # Perform PCA and visualize for original video
    original_em_values = np.vstack(original_embeddings)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(original_em_values)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=original_time_seconds)
    plt.colorbar(label='Time (in seconds)')
    plt.title('Example 1: Original video frame embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.show()

    # Perform PCA and visualize for keyframes
    key_em_values = np.vstack(key_embeddings)
    pca = PCA(n_components=2)
    key_reduced_embeddings = pca.fit_transform(key_em_values)
    plt.scatter(key_reduced_embeddings[:, 0], key_reduced_embeddings[:, 1], c=key_time_seconds)
    plt.colorbar(label='Time (in seconds)')
    plt.title('Example 2: Keyframe video frame embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.show()

def is_sequence(lst):
    return all(lst[i] + 1 == lst[i+1] for i in range(len(lst) - 1))

def plot_tsne_and_centroids(reduced_embeddings, indices):
    plt.figure(figsize=(10, 5))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], label='Frames')
    centroids = [np.mean(reduced_embeddings[idx], axis=0) for idx in indices]
    for i, (x, y) in enumerate(reduced_embeddings):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='red', label='Centroids')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title('Centroid-Based Sequence Analysis of Frame Embeddings')
    plt.show()

def plot_frames(indices, frame_embedding_pairs, reduced_embeddings, max_rows=1, save_dir=None):
    plotted_indices = set()
    sequence_count = defaultdict(int)
    unique_frames = []
    labels = []
    for i, neighbors in enumerate(indices):
        knn_cluster = reduced_embeddings[neighbors]
        centroid = tuple(np.mean(knn_cluster, axis=0))
        sequence_count[centroid] += is_sequence(sorted(neighbors))
    sorted_centroids = sorted(sequence_count.keys(), key=lambda x: (sequence_count[x], np.linalg.norm(np.array(x))), reverse=True)
    for centroid in sorted_centroids:
        subset_indices = [i for i, neighbors in enumerate(indices) if tuple(np.mean(reduced_embeddings[neighbors], axis=0)) == centroid]
        unique_subset_indices = [idx for idx in subset_indices if idx not in plotted_indices]
        for emb_idx in unique_subset_indices:
            frame, _, _ = frame_embedding_pairs[emb_idx]
            unique_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            distance_to_centroid = np.linalg.norm(reduced_embeddings[emb_idx] - np.array(centroid))
            labels.append(f"Frame {emb_idx}\nCentroid Seq Count: {sequence_count[centroid]}\nDistance: {distance_to_centroid:.2f}\nCentroid: ({centroid})")
            plotted_indices.add(emb_idx)

    if len(unique_frames) > 0:
        total_rows = int(np.ceil(len(unique_frames) / 4.0))
        fig, axes = plt.subplots(total_rows, 4, figsize=(10, 5 * total_rows / 4))
        if total_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for i, ax in enumerate(axes.flatten()):
            if i >= len(unique_frames):
                ax.axis('off')
                ax.set_visible(False)
                continue
            ax.imshow(unique_frames[i])
            ax.axis('off')
            ax.set_title(labels[i], fontsize=5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        print("No unique frames to plot.")

def run_knn(knn=6, embedding_values=None, frame_embedding_pairs=None, num_samples=None):
    # Randomly sample indices if num_samples is provided
    if num_samples is not None:
        total_samples = len(embedding_values)
        if num_samples > total_samples:
            print("Number of samples exceeds the total number of embeddings. Using all embeddings.")
            sampled_indices = np.arange(total_samples)
        else:
            sampled_indices = np.random.choice(total_samples, num_samples, replace=False)
        embedding_values = np.array([embedding_values[i] for i in sampled_indices])
        frame_embedding_pairs = [frame_embedding_pairs[i] for i in sampled_indices]
    # Use KNN to find closest neighbors
    knn_model = NearestNeighbors(n_neighbors=knn)
    knn_model.fit(embedding_values)
    indices = knn_model.kneighbors(embedding_values, return_distance=False)
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embedding_values)
    plot_tsne_and_centroids(reduced_embeddings, indices)
    plot_frames(indices, frame_embedding_pairs, reduced_embeddings, max_rows=4, save_dir=None)

def run_optimized_knn(embedding_values, clip_indices_to_visualize=None, scatter_size=(10, 10), random_sample=None):
    """Run KNN optimization and t-SNE on embeddings and plot the results."""
    # If clip_indices_to_visualize is None, consider all indices
    if clip_indices_to_visualize is None:
        clip_indices_to_visualize = list(range(len(embedding_values)))
    # If random_sample is not None, randomly sample that many indices
    if random_sample is not None:
        clip_indices_to_visualize = np.random.choice(clip_indices_to_visualize, random_sample, replace=False)
    # Filter the embedding_values to only include indices in clip_indices_to_visualize
    filtered_embedding_values = np.array([embedding_values[i] for i in clip_indices_to_visualize])
    # Initialize KNN model
    knn = NearestNeighbors()
    # Fit the KNN model to the filtered data
    knn.fit(filtered_embedding_values)
    # Initialize variables to keep track of the best k and performance
    max_clusters = 0
    best_k = 0
    k_performance = []
    # Determine the maximum k value to consider
    max_k = min(25, len(filtered_embedding_values))
    # Loop through different k values to find the one that maximizes sequence clusters
    for k in range(5, max_k + 1):
        indices = knn.kneighbors(filtered_embedding_values, n_neighbors=k, return_distance=False)
        sequence_clusters = sum(is_sequence(sorted(neighbors)) for neighbors in indices)
        k_performance.append(sequence_clusters)
        
        if sequence_clusters > max_clusters:
            max_clusters = sequence_clusters
            best_k = k
    # Plot the performance over different k values
    plt.figure()
    plt.plot(range(5, max_k + 1), k_performance)
    plt.xlabel('k')
    plt.ylabel('Number of Sequence Clusters')
    plt.title('Sequence to K Analysis')
    plt.show()
    # Perform t-SNE dimensionality reduction
    dynamic_perplexity = min(30, len(filtered_embedding_values) - 1)
    tsne = TSNE(n_components=2, perplexity=dynamic_perplexity)
    reduced_embeddings = tsne.fit_transform(filtered_embedding_values)
    # Get the k nearest neighbors using the best k
    indices = knn.kneighbors(filtered_embedding_values, n_neighbors=best_k, return_distance=False)
    # Plot the t-SNE reduced data
    plt.figure(figsize=scatter_size)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], label='Frames')
    centroids = [np.mean(reduced_embeddings[idx], axis=0) for idx in indices]
    for i, (x, y) in enumerate(reduced_embeddings):
        distance = np.linalg.norm([x, y] - centroids[i])
        plt.annotate(f"{distance:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='red', label='Centroids')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title(f't-SNE Scatter Plot with Best k = {best_k}')
    plt.show()


def visualize_clips(num_clips, vid_cap, embedding_values, window_size=6, threshold_factor=0.5):
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(vid_cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize variables
    rolling_window = []
    frames_to_plot = []
    
    # Specify clip indices to visualize
    clip_indices_to_visualize = list(range(0, min(num_clips, len(embedding_values))))
    next_indices = clip_indices_to_visualize[1:] + [None]
    for i, emb_idx in enumerate(clip_indices_to_visualize):
        start_frame = emb_idx 
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        success, frame = vid_cap.read()
        if not success:
            break
        next_idx = next_indices[i]
        if next_idx is not None:
            similarity = euclidean_distances(embedding_values[emb_idx].reshape(1, -1), embedding_values[next_idx].reshape(1, -1))[0][0]
            rolling_window.append(similarity)
        else:
            similarity = None
        if len(rolling_window) > window_size:
            rolling_window.pop(0)
        avg_similarity = np.mean([x for x in rolling_window if x is not None])
        if similarity is not None and similarity > avg_similarity * threshold_factor:
            print(f"Potential visual transition at frame {start_frame} with similarity score {similarity}")
            frames_to_plot.append((frame, start_frame, np.mean(embedding_values[emb_idx]), similarity))
    
    # Plotting
    n_rows = int(np.ceil(len(frames_to_plot) / 5))
    fig = plt.figure(figsize=(20, 4 * n_rows))
    for i, (frame, start_frame, avg_embedding, similarity) in enumerate(frames_to_plot):
        ax = fig.add_subplot(n_rows, 5, i+1)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Frame {start_frame}\nAvg Embedding: {avg_embedding}\nSuccessor Similarity: {similarity}', fontsize=10)

    plt.show()
    vid_cap.release()