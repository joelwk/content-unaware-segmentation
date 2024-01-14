import os
import glob
import pandas as pd
import cv2
import numpy as np
from clip_video_encode import clip_video_encode
from pipeline import read_config

directories = read_config(section="directory")

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return duration

def get_average_embedding(emb_path):
    embeddings = np.load(emb_path)
    average_embedding = np.mean(embeddings, axis=0)
    return average_embedding

def clip_encode(parquet_path, emb_folder):
    if os.path.exists(parquet_path):
        clip_video_encode(
            parquet_path,
            emb_folder,
            frame_workers=25,
            take_every_nth=1,
            metadata_columns=['videoLoc', 'videoID', 'duration']
        )
    else:
        print(f"Parquet file {parquet_path} does not exist. Skipping clip_video_encode.")

def process_videos_and_metadata(dataset_folder, emb_folder):
    for sub_folder in glob.glob(os.path.join(dataset_folder, '*')):
        video_files = glob.glob(os.path.join(sub_folder, '*.mp4'))
        clips = []
        new_id = 0  # Initialize separate ID tracker
        for index, video_file_path in enumerate(video_files):
            duration = get_video_duration(video_file_path)
            if duration is None:
                continue
            metadata = {
                "videoID": str(new_id),
                "videoLoc": video_file_path,
                "duration": duration
            }
            clips.append(metadata)
            new_id += 1  # Increment ID tracker

        clips_df = pd.DataFrame(clips)
        
        # Check for empty DataFrame and skip if necessary
        if clips_df.empty:
            print(f"No clips found for {sub_folder}. Skipping.")
            continue

        updated_parquet = os.path.join(sub_folder, '00000.parquet')
        if not os.path.exists(updated_parquet):
            clips_df.to_parquet(updated_parquet, index=False)
        clip_encode(updated_parquet, emb_folder)
        
        average_embeddings = {}
        for index, row in clips_df.iterrows():
            emb_file_path = os.path.join(emb_folder, f"{str(index)}.npy")
            json_file_path = os.path.join(emb_folder, f"{str(index)}.json")
            
            # Check for existing embedding files
            if not os.path.exists(emb_file_path):
                print(f"No embedding found for video {str(index)}. Skipping.")
                continue
                
            average_embedding = get_average_embedding(emb_file_path)
            average_embeddings[f"{str(index)}.mp4"] = average_embedding
            
            avg_emb_folder = os.path.join(emb_folder, os.path.basename(sub_folder))
            os.makedirs(avg_emb_folder, exist_ok=True)
            
            average_emb_file_path = os.path.join(avg_emb_folder, f"{str(index)}_average.npy")
            np.save(average_emb_file_path, average_embedding)
            
            # Move the corresponding JSON files
            new_json_file_path = os.path.join(avg_emb_folder, f"{str(index)}.json")
            os.rename(json_file_path, new_json_file_path)
            
            # Remove the original embedding file to clean up
            os.remove(emb_file_path)

def main():
    dataset_folder = os.path.join(directories['base_directory'], directories['keyframe_clip_output'])
    emb_folder = os.path.join(directories['base_directory'], directories['keyframe_clip_embeddings_output'])
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(emb_folder, exist_ok=True)
    process_videos_and_metadata(dataset_folder, emb_folder)
if __name__ == "__main__":
    main()


