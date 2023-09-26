# Project Description
This project introduces an efficient approach to video segmentation aimed at summarizing videos into highly representative key moments. The core technique involves analyzing semantic embeddings extracted from video frame embedding pairs. Each frame is assigned a score calculated based on the Euclidean distance between the embedding of the current frame and its successor. A significant distance implies dissimilarity between frames, serving as a marker for a potential transition or "seam" between segments. These seams are identified when the successor score surpasses a dynamically adjusted threshold, or set according to a predetermined value. The system is also capable of adapting to changes in the video content by using a rolling average and standard deviation to adjust the threshold. The system is evaluated on a variety of videos, including those with multiple segments, and is shown to be effective at identifying key moments. 

## Key features:
1. **Dynamic Thresholds**: Adapts to varying video content using a rolling average and standard deviation to adjust the threshold.
2. **Embedding Semantics**: Using semantic embeddings allows for a rich representation of content, enabling the system to identify better segments where a significant change occurs.
3. **Successor Score**: The primary heuristic in keyframe detection through the Euclidean distance between successive frame embeddings.
4. **Embedding Surveyor**: Leverages K-Nearest Neighbors (KNN) to fine-tune the dynamic thresholds, providing a second layer of adaptability and increasing segmentation accuracy.
5. **Seam Detection**: Leveraging the successor score and KNN for detecting "trending" seams presents a novel way of identifying key moments without needing explicit object recognition or manual labeling.
6. **Adaptive System**: The combination of dynamic thresholds and successor scores allows the system to adapt to different videos and changes within a single video.

## Datasets (`datasets/`)
## Processing (`processing/`)
## Primary Scripts (`scripts/`)
`embedding_surveyer/` - Unsupervised video segmentation using successor embeddings and distance thresholding. 
## Evaluation Plots (`plots/`)
## Primary Examples (`notebooks/`)
## Configuration and Setup (`config.ini`)
The following is a list of the inital parameters and their default values: 
- Directories
    - datasets = ./datasets/originalvideos
    - keyframes = ./datasets/originalkeyframes
    - embeddings = ./datasets/originalembeddings
- Thresholds
    - avg_distance = 0.5
    - std_dev = 0.15
    - successor_value = 0.6
    - window_size = 30
    - step_size = 15
## Collection and Processing
Primary Examples (`examples/`)
## Future Work

## References
Su, J., Yin, R., Zhang, S., & Luo, J. (2023). Motion-state Alignment for Video Semantic Segmentation.
Cho, S., Kim, W. J., Cho, M., Lee, S., Lee, M., Park, C., & Lee, S. (2022). Pixel-Level Equalized Matching for Video Object Segmentation
Han, Z., He, X., Tang, M., & Lv, Y. (2021). Video Similarity and Alignment Learning on Partial Video Copy Detection
Cho, D., Hong, S., Kang, S., & Kim, J. (2019). Key Instance Selection for Unsupervised Video Object Segmentation