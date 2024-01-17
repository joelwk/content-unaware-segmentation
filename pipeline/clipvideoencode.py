from pipeline import read_config, install_local_package, save_metadata_to_parquet
import load_data as ld
import glob
import subprocess
import argparse
import os
from contextlib import contextmanager
import webdataset as wds
import json
import numpy as np
import io

directories = read_config(section="directory")
evaluations = read_config(section="evaluations")
config_params = read_config(section="config_params")
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
clipencode_abs_path = os.path.join(base_path,'pipeline', 'clip-video-encode')

def collect_video_metadata(video_ids, json_metadata, output):
    keyframe_video_locs = []
    original_video_locs = []


    for video_id, metadata in zip(video_ids, json_metadata):
        print(f"Loaded metadata for {video_id}: {metadata}")

        # Fallback to get_video_duration if 'duration' is not available in JSON metadata
        duration = metadata.get('video_metadata', {}).get('streams', [{}])[0].get('duration', None)
        if duration is None:
            print(f"Duration not found in metadata. Calculating duration for {video_id}.")
            video_file_path = os.path.join(output, f"{video_id}.mp4")  # Assuming video file path
            duration = ld.get_video_duration(video_file_path)

        keyframe_video_locs.append({
            "videoLoc": f"{output}/{video_id}_key_frames.mp4",
            "videoID": video_id,
            "duration": duration,
        })
        original_video_locs.append({
            "videoLoc": os.path.join(output, f"{video_id}.mp4"),
            "videoID": video_id,
            "duration": duration,
        })

    return keyframe_video_locs, original_video_locs

@contextmanager
def change_directory(destination):
    original_path = os.getcwd()
    if not os.path.exists(destination):
        os.makedirs(destination)
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(original_path)

def extract_video_id(key):
    parts = key.split('/')
    if len(parts) > 1:
        return parts[1].split('_')[0]
    return None

def clip_encode():
    base_directory = directories['base_directory']
    if directories['video_load'] == 'download' and config_params['mode'] == 'directory':
        embeddings = os.path.join(base_directory, directories['embeddings'])
        with change_directory(clipencode_abs_path):
            from clip_video_encode import clip_video_encode
        clip_video_encode(
                f'{directories["base_directory"]}/keyframe_video_requirements.parquet',
                    embeddings,
                    frame_workers=int(directories['frame_workers']),
                    take_every_nth=int(directories['take_every_nth']),
                    metadata_columns=['videoLoc', 'videoID', 'duration'])
                    
    elif directories['video_load'] == 'directory' and config_params['mode'] == 'wds':
        dataset_paths = glob.glob(f"{evaluations['wds_dir']}")
        dataset = wds.WebDataset(dataset_paths).map(ld.process_files)
        for sample in dataset:
            print(sample)
            key = sample['__key__']
            video_id = extract_video_id(key)
            print(video_id)
            if metadata:
                keyframe_video_locs, original_video_locs = collect_video_metadata([video_id], [metadata], directories['base_directory'])
                save_metadata_to_parquet(keyframe_video_locs, original_video_locs, directories['base_directory'])
                embedding_file = sample.get('embedding_file')
                if embedding_file:
                    with change_directory(clipencode_abs_path):
                        from clip_video_encode import clip_video_encode
                        clip_video_encode(
                            f'{directories["base_directory"]}/keyframe_video_requirements.parquet',
                            embedding_file,
                            frame_workers=int(directories['frame_workers']),
                            take_every_nth=int(directories['take_every_nth']),
                            metadata_columns=['videoLoc', 'videoID', 'duration'])
                            
def main():
    clip_encode()
    return 0
if __name__ == "__main__":
    main()