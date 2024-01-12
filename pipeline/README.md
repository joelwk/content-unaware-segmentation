# Sucessor Segmentation with CLIP


Segment video content with CLIP embeddings using successor frame disimilarity 

## Setup and Run
#### Clone the repo and load data using one of the options below

1. Modify json in `./content-unaware-segmentation/pipeline/pipeline.py`
```
    dataset_requirements = {
        "data": [
          {"url": "www.youtube.com/watch?v=iqrpwARx26E", 
          "caption": "Elon Musk on politics: I would not vote for a pro-censorship candidate"},
          {"url": "www.youtube.com/watch?v=YEUclZdj_Sc", 
          "caption": "Why next-token prediction is enough for AGI"},
        ]
      }
```

2. Modify config.ini in `./content-unaware-segmentation/pipeline/config.ini` and  replace `external_parquet = None` with the parquet directory.
  - `url` will be a youtube link or directory location
  - `caption` can be the youtube title or short video description

#### Reference the Config details below and run the following commands when ready:
```
# To run the segmentation pipeline:
python ./clip-video-encode/examples/successor_segmentation/run_segmentation.py

# To run the evaluation pipeline:
python ./clip-video-encode/examples/successor_segmentation/run_zeroshots.py
```
## Download videos and segment
If you want to download the videos and segment them, then make the following changes:
1. Set `video_load` to download 
2. Leave `base_directory` as is or change to your own directory

## Load videos and segment
If the videos are already downloaded then make the following changes:
1. Set `video_load` to directory 
2. Create a dataset_requirements.parquet file with the following columns: `url` and `caption`- see above for full details. 
2. Update `base_directory` and `originalframes` to the directory that contains the videos and dataset_requirements.parquet made in step 2

## Config details
```
[directory] - (all variables are either directory path or None)
base_directory: main dataset directory
originalframes: original video directory
keyframes: keyframe (images) of original video
embeddings: embeddings of keyframe video
originalembeddings: original embeddings of original video
keyframe_clip_embeddings_outputs: keyframe clip (video embeddings) embeddings
output: temp directory before copying to ./completed
cut_segments_outputs: temp directory before copying to ./completed
keyframe_outputs: temp directory before copying to ./completed
keyframe_clips_output: temp directory before copying to ./completed
keyframe_audio_clip_output: temp directory before copying to ./completed
video_wds_output: where the wds will be saved
external_parquet: location of external parquet file with list of video directories or youtube links
frame_workers: int: number of Processes to distribute video reading to
take_every_nth: int: only take every nth frame
video_load: str: directory, if the videos are already downloaded using yt-dl, download, if not, then the videos will be downloaded

[config_params]
mode: str: used for evaluation, if wds, then the a WebDataset will be used to evaluate videos, if directory, then the completedatasets directory(./completedatasets) will be used
full_whisper_audio: bool: if True, then the full audio will be transcribed, if False, then only the audio segmennts will be transcribed
transcript_mode: str: if all, both whisper transcripts and youtube subtitles are created, if whisper, then only whisper transcripts are created, if yt, then only youtube subtitles are created
segment_video: bool: if True, then the video will be segmented, if False, then the video will not be segmented
segment_audio: bool: if True, then the audio will be segmented, if False, then the audio will not be segmented
compute_embeddings: bool: if True, then the average embedding value of each segment will be computed, if False, then the average embedding value of each segment will not be computed
specific_videos: None or list of int in order of videos in parquet: if None, then all videos in the dataset will be processed, if list, then only the videos in the list will be processed
plot_grid: bool: if True, then a grid of the keyframes will be plotted and saved, if False, then a grid of the keyframes will not be plotted and saved

[thresholds]
max_duration: int: max duration in seconds that each segment can be
keyframe_audio_duration: int: max duration that each keyframe will be in seconds
avg_distance: float: distance between adjacent frames
std_dev: float: std distance between adjacent frames
successor_value: float: threshold of successor value, indicates a keyframe if exceeded
window_size: int: sliding window size for distance calculation
step_size: int: sliding window step size for distance calculation
tolerance: float: if audio lags video, reduce or increase accordingly
phash_threshold: float: threshold for phash similarity
is_person_threshold - float: threshold for person detection 
single_face_threshold: float: threshold for single face detection
facing_forward_threshold: float: threshold direction of face
engagement_threshold: float: threshold for persons actions 
type_person_threshold: float: threshold for gender and age cohort

[evaluations]
pipeline_function: str: transformer pipeline operation to perform (default: automatic-speech-recognition)
chunk_length: int: number of chunks to divide the audio into
batch_size: int: number of chunks to process at once
whisper_model: str: whisper model to use (default: openai/whisper-large-v2)
model_clip: str: CLIP model to use (default: hf-hub:apple/DFN5B-CLIP-ViT-H-14-378)
model_clap_checkpoint: if not None, then the model will be loaded from the checkpoint
model_clap: str: CLAP model to use (default: HTSAT-base)
scalingfactor: int: scaling multiple for softmax output
max_duration_ms: int: number of milliseconds for each whisper segment
wds_dir: str: directory to save the WebDataset
embeddings: str: label embeddings
outputs: str: directory of final evaluation output
completedatasets: str: directory of final segmentation output
audio_threshold: float:  threshold for audio classification
face_detected_in_video_or: bool: if True, then all keyframes are classified, if False, only face detected keyframes are classified
```

## Documentation and Examples

### Walk-through and Colab notebook
* [Pipeline demo](https://colab.research.google.com/drive/1ZYAczt1sfXCbsakgr5dmgFejLNmcLMPB?usp=sharing)

### Segmentation process
* [content-unaware-segmentation](https://github.com/joelwk/content-unaware-segmentation) -  An efficient approach to video segmentation aimed at summarizing videos into highly representative keyframes. The core technique involves analyzing semantic embeddings extracted from video frame embedding pairs. Each frame is assigned a score calculated based on the Euclidean distance between the embedding of the current frame and its successor.

- [clip-video-encode](https://github.com/iejMac/clip-video-encode/blob/main/clip_video_encode/clip_video_enc)


```bibtex
@misc{kilian-2023-video2dataset,
  author = {Maciej Kilian, Romain Beaumont, Daniel Mendelevitch, Sumith Kulal, Andreas Blattmann},
  title = {video2dataset: Easily turn large sets of video urls to a video dataset},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/iejMac/video2dataset}}
}