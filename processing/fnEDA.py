import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

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

def plot_embedding_and_similarity(fig, gs, frame_indices, embedding_values):
    ax1 = plt.subplot(gs[-5:])
    plot_indices = frame_indices
    next_indices = plot_indices[1:] + [None]
    
    ax1.bar(range(len(plot_indices)), [euclidean_distances(embedding_values[i].reshape(1, -1),
                                                           embedding_values[next_indices[i]].reshape(1, -1))[0][0] if next_indices[i] is not None else 0 for i, _ in enumerate(plot_indices)], color='green', label='Successor Frame Similarity')
    ax1.set_xticks(range(len(plot_indices)))
    ax1.set_xticklabels(plot_indices)
    ax1.set_xlabel('Frame Embedding Index', fontsize=10)
    ax1.set_ylabel('Similarity', fontsize=10)

    ax2 = ax1.twinx()
    avg_embedding_values = [np.mean(embedding_values[i]) for i in plot_indices if i < len(embedding_values)]
    ax2.scatter(range(len(plot_indices)), avg_embedding_values, color='red', label='Avg. Embedding Vectors')
    ax2.set_ylabel('Avg. Embedding Vector', fontsize=10)

    for i, txt in enumerate(avg_embedding_values):
        ax2.annotate(f"{txt:.6f}", (i, txt), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Successor Similarity & Avg. Embedding Vectors Over Video Duration', fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

if __name__ == "__main__":
    key_video_files = None
    vid_cap, frame_rate = initialize_video(key_video_files[0])
    frame_indices_to_visualize = list(range(0, 10))
    
    # Make sure to initialize embedding_values somewhere in your code
    
    fig, gs = visualize_frames(vid_cap, frame_indices_to_visualize, embedding_values)  # Collect fig
    plot_embedding_and_similarity(fig, gs, frame_indices_to_visualize, embedding_values)  # Pass fig
    vid_cap.release()
