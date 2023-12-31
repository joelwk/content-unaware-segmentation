# Project Description
This project introduces an efficient approach to video segmentation aimed at summarizing videos into highly representative keyframes. The core technique involves analyzing semantic embeddings extracted from video frame embedding pairs. Each frame is assigned a score calculated based on the Euclidean distance between the embedding of the current frame and its successor. A significant distance implies dissimilarity between frames as a marker for a potential transition or "seam" between segments. These seams are identified when the successor score surpasses a dynamically adjusted threshold or is set according to a predetermined value. 
##


<div style="margin: 0 auto; width: 80%;">
  <table style="margin: 0 auto;">
    <tr>
      <td colspan="2" style="text-align: center;">
        <a href="https://www.youtube.com/watch?v=nXBoOam5xJs">The Deadly Portuguese Man O' War | 4KUHD | Blue Planet II | BBC Earth</a>
        <br>
        <a href="./plots/keyframes_grid.png" target="_blank">Link to full keyframe grid for video 1</a>
      </td>
    </tr>
      <td style="text-align: center;">
        <img src="./plots/original_video_scatter_1.png" alt="Original Video Scatter Plot" width="350" height="250">
      </td>
      <td style="text-align: center;">
        <img src="./plots/key_video_scatter_1.png" alt="Key Video Scatter Plot" width="350" height="250">
      </td>
    </tr>
    <tr>
      <td rowspan="2" style="text-align: center;">
        <img src="./plots/original_video_embeddings_1.png" alt="Original Video Embeddings" width="350" height="325"><br>
        <img src="./plots/keyframe_embeddings_1.png" alt="Keyframe Embeddings" width="350" height="325">
      </td>
      <td style="text-align: center;">
        <img src="./plots/1.png" alt="Alt text" width="350" height="650">
      </td>
    </tr>
  </table>
</div>

## Summary Review of Process
1. **Download YouTube Video and Extract Keyframes Using video2dataset**
    - `video2dataset` employs `yt-dl` for video downloading and `ffmpeg` for keyframe extraction.
    - This step yields: original videos, keyframes, and the YouTube metadata JSON.

2. **Convert Original Frames and Keyframes to Numpy Array**
    - The original frames and keyframes are processed via `clip-video-encode` to generate their embedding vectors.
    - These vectors are created using a pre-trained `laion2b_s34b_b79k ViT-B-32` model.
    - The embeddings are a compact representation of the frames and keyframes, encapsulating essential visual features.

3. **Analysis, Visualization, and Fine-tuning**
    - A sliding window method and k-NN are used to identify segments where the successor value crosses a specific threshold or shows a unique pattern.
    - After determining optimal values, the script is configured to produce image and video keyframes.
    - The primary output consists of 2-15-second clips, each containing 2-4 keyframes most representative of the clip's content.

    <summary>Key Terms</summary>

      - **Distance Metrics**: Uses Euclidean distance to measure the similarity between embeddings of adjacent frames.
      - **Successor Values**: The Euclidean distance of the current frame to its successor frame - used to qualify new segments.
    </details>

## Key features:
1. **Dynamic Thresholds**: Adapts to varying video content using a rolling average and standard deviation to adjust the threshold.
2. **Embedding Semantics**: Using semantic embeddings allows for a rich representation of content, enabling the system to identify better segments where a significant change occurs.
3. **Successor Score**: The primary heuristic in keyframe detection is through the Euclidean distance between successive frame embeddings.
4. **Embedding Surveyor**: Leverages K-Nearest Neighbors (KNN) to fine-tune the dynamic thresholds, providing a second layer of adaptability and increasing segmentation accuracy.
5. **Seam Detection**: Leveraging the successor score and KNN for detecting "trending" seams presents a novel way of identifying key moments without needing explicit object recognition or manual labeling.
6. **Adaptive System**: The combination of dynamic thresholds and successor scores allows the system to adapt to different videos and changes within a single video.

## File Structure and Descriptions
### `dataset/`
- `evaluations/`: Embedding summary statistics by video
- `keyframeembeddings/`: Keyframe embeddings
- `keyframes/`: Inital video keyframes, reduced FPS version of original video 
- `originalembeddings/`: Full video embeddings
- `originalvideos/`: Full videos used to create keyframes
### `notebooks/`
- `evaluations/`: Embedding statistics by video
- `EDA.ipynb`: Inital EDA and methods visualized
- `Example1.ipynb`: Embedding surveyor with video 1
- `Example2.ipynb`: Embedding surveyor with video 7
- `Results EDA.ipynb`: Successor segmentation video 1
- `Generative Summarization EDA.ipynb`: Final summarization process - **work in progress**
### `pipeline/`
- [`Successor-Segmentation-Pipeline.ipynb`](https://colab.research.google.com/drive/1ZYAczt1sfXCbsakgr5dmgFejLNmcLMPB?usp=sharing): Google Colab notebook for full video segmentation pipeline.
- `pipeline.py`: Setup script for running the pipeline.
- `clipvideoencode.py`: Script for extracting embeddings from video frames - uses `clip-video-encode` library.
- `video2dataset.py`: Script for downloading YouTube videos with metadata and extracting keyframes - uses `video2dataset` library.
- `segment_averaging.py`: Script for calculating average clip embedding for each cut segment.
- `move_and_group.py` & `rename_and_move_files.py`: Utility scripts for organizing files.
### `plots/`
- `1.png through 8.png`: Grid plots of the keyframes for each video.
- `original_video_scatter_1.png` & `key_video_scatter_1.png`: Scatter plots for video 1 for original and keyframe embeddings - shows the latent space for each video.
- `original_video_embeddings_1` & `original_video_embeddings_7`: Visualize the relationship between video frames and embedding values over video duration. Video frames are plotted with associated histograms of average embedding values and successor similarity scores.
- `keyframe_embeddings_1` & `keyframe_embeddings_7`: Visualize the relationship between keyframes and keyframe embedding values over video duration. Video frames are plotted with associated histograms of average embedding values and successor similarity scores.
- [Example outputs](https://drive.google.com/drive/folders/1Z2gldAViNL44Y7j2U893d_2wLPuvvEKT?usp=sharing)
### `processing/`
- `segment_processing.py`: Provides utility functions for video segmentation and keyframe filtering based on metrics like perceptual hashing and Euclidean distances between embeddings. It reads configurable thresholds and uses them to detect new segments in a video, filter out similar keyframes, and calculate distances to centroids.
### `scripts/`
- `embedding_surveyor.py`: The SlidingWindowAnalyzer class in Python is designed for video segmentation and keyframe analysis. It uses a sliding window approach to analyze video embeddings, employing algorithms like K-Nearest Neighbors (KNN) and t-SNE for clustering and visualization. The class also dynamically updates distance thresholds and leverages various utility functions for plotting and threshold management tasks.
- `successor_segmentation.py`: This file contains the SegmentSuccessorAnalyzer class, designed for video keyframe analysis and segmentation. It operates on pre-computed video embeddings and uses configurable thresholds for segment identification. The class also incorporates optional maximum segment duration constraints and saves keyframes and their metadata for further analysis. It is part of the pipeline that can be run on multiple videos, and it leverages utility functions for tasks like annotation and plotting.
- `fold_seams.py`: The primary function in this file is `segment_video_using_keyframes_and_embeddings` and is designed to segment a video based on keyframe timestamps (obtained from `successor_segmentation.py`). It uses FFmpeg for the actual video cutting. The function also incorporates a tolerance level that can be adjusted to fine-tune each segment's start and end times. This tolerance is mainly used when the segmentation is based on keyframes so that each segment's start and end times are not too close to the keyframe timestamps.


## Relevant Literature and Citations
- Su, J., Yin, R., Zhang, S., & Luo, J. (2023). Motion-state Alignment for Video Semantic Segmentation.
- Cho, S., Kim, W. J., Cho, M., Lee, S., Lee, M., Park, C., & Lee, S. (2022). Pixel-Level Equalized Matching for Video Object Segmentation.
- Han, Z., He, X., Tang, M., & Lv, Y. (2021). Video Similarity and Alignment Learning on Partial Video Copy Detection.
- Cho, D., Hong, S., Kang, S., & Kim, J. (2019). Key Instance Selection for Unsupervised Video Object Segmentation.
- Foster, D. (n.d.-b). Generative Deep Learning, 2nd Edition. O'Reilly Online Learning. [O'Reilly Online Learning](https://learning.oreilly.com/library/view/generative-deep-learning/9781098134174/ch03.html)

### Relevant Tools and Libraries
- [clip-video-encode](https://github.com/iejMac/clip-video-encode/blob/main/clip_video_encode/clip_video_encode.py)

```bibtex
@misc{kilian-2023-video2dataset,
  author = {Maciej Kilian, Romain Beaumont, Daniel Mendelevitch, Sumith Kulal, Andreas Blattmann},
  title = {video2dataset: Easily turn large sets of video urls to a video dataset},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/iejMac/video2dataset}}
}
