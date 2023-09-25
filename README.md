# Project Description
This project introduces an efficient approach to video segmentation aimed at summarizing videos into highly representative key moments. The core technique involves analyzing semantic embeddings extracted from video frame embedding pairs. Each frame is assigned a score calculated based on the Euclidean distance between the embedding of the current frame and its successor. A significant distance implies dissimilarity between frames, serving as a marker for a potential transition or "seam" between segments. These seams are identified when the successor score surpasses a dynamically adjusted threshold, bringing adaptability.

## Key features:
1. **Dynamic Thresholds**: The system adapts to varying video content by using a rolling average and standard deviation to adjust the threshold dynamically.

2. **Embedding Semantics**: Utilization of semantic embeddings allows for a richer representation of content, enabling the system to better identify segments where a significant change occurs.

3. **Successor Score**: A unique approach of using the Euclidean distance between successive frame embeddings to calculate a "successor score," which aids in the detection of video segments.

4. **Embedding Surveyor**: Leverages K-Nearest Neighbors (KNN) to fine-tune the dynamic thresholds, providing a second layer of adaptability and increasing segmentation accuracy.

5. **Novel Seam Detection**: Leveraging the successor score and Embedding Surveyor for detecting seams presents a novel way of identifying key moments without the need for explicit object recognition or manual labeling.

6. **Adaptive System**: The combination of dynamic thresholds, successor scores, and Embedding Surveyor allows the system to adapt not just to different videos but also to changes within a single video.

7. **Real-time Analysis**: Designed for efficiency, making it feasible for real-time video analysis.


## Datasets (`datasets/`)
## Processing (`processing/`)
## Primary Scripts (`scripts/`)
## Evaluation Plots (`plots/`)
## Primary Examples (`notebooks/`)
## Configuration and Setup (`config.ini`)
The configuration file is used to set the inital conditions. The following is a list of the parameters and their default values: 
- Directories
    - datasets = ./datasets/originalvideos
    - keyframes = ./datasets/originalkeyframes
    - embeddings = ./datasets/originalembeddings
- Thresholds
    - successor_value = 0.6
    - window_size = 30
    - step_size = 15
## Collection and Processing
Primary Examples (`examples/`)
## Future Work

## References
