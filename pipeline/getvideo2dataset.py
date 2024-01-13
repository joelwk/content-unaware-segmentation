import os
import pandas as pd
import json
import glob
import configparser
import subprocess
import argparse
import shutil
import sys
import ffmpeg
from pipeline import read_config, generate_config, install_local_package, parse_args
from load_data import get_video_duration
import cv2

def load_dataset_requirements(directory):
    return pd.read_parquet(f"{directory}/dataset_requirements.parquet").to_dict(orient='records')

def collect_video_metadata(video_files, output):
    keyframe_video_locs = []
    original_video_locs = []
    for video_file in video_files:
        video_id = os.path.basename(video_file).split('.')[0]
        json_meta_path = video_file.replace('.mp4', '.json')
        if not os.path.exists(json_meta_path):
            print(f"JSON metadata file does not exist: {json_meta_path}")
            continue
        with open(json_meta_path, 'r') as f:
            metadata = json.load(f)
            print(metadata)
        print(f"Loaded metadata for {video_id}: {metadata}")
            
        # Fallback to get_video_duration if 'duration' is not available in JSON metadata
        duration = metadata.get('video_metadata', {}).get('streams', [{}])[0].get('duration', None)
        print(duration)
        if duration is None:
            print(f"Duration not found in metadata. Calculating duration for {video_id}.")
            duration = get_video_duration(video_file)
        print(keyframe_video_locs)
            
        keyframe_video_locs.append({
            "videoLoc": f"{output}/{video_id}_key_frames.mp4",
            "videoID": video_id,
            "duration": duration,
        })
        original_video_locs.append({
            "videoLoc": video_file,
            "videoID": video_id,
            "duration": duration,
        })
    return keyframe_video_locs, original_video_locs

def fix_codecs_in_directory(directories):
    base_directory = directories['base_directory']
    video_files = glob.glob(os.path.join(base_directory, directories["original_frames"], '**/*.mp4'), recursive=True)
    print(video_files)
    for video_file in video_files:
        video_id = os.path.basename(video_file).split('.')[0]
        input_file_path = video_file 
        output_file_path = os.path.join(base_directory, directories["original_frames"], f"fixed_{video_id}.mp4")
        try:
            ffmpeg.input(input_file_path).output(output_file_path, vcodec='libx264', strict='-2', loglevel="quiet").overwrite_output().run(capture_stdout=True, capture_stderr=True)
            print(f"Successfully re-encoded {video_file}")
            os.remove(input_file_path)
            os.rename(output_file_path, input_file_path)
        except AttributeError as e:
            print(f"AttributeError: {e}. FFMPEG might not be correctly installed or imported.")
        except Exception as e:  
            print(f"An unexpected error occurred: {e}")

def segment_key_frames_in_directory(directories):
    base_directory = directories['base_directory']
    video_files = glob.glob(os.path.join(base_directory, directories["original_frames"], '**/*.mp4'), recursive=True)
    for video_file in video_files:
        video_id = os.path.basename(video_file).split('.')[0]
        output_file = os.path.join(base_directory, directories["keyframes"], f"{video_id}_key_frames.mp4")
        print(f"Segmenting key frames for {video_id}...")
        command = f'ffmpeg -y -loglevel error -discard nokey -i {video_file} -c:s copy -c copy -copyts {output_file}'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print(f"Successfully filtered best frames for {video_id}.")
        else:
            print(f"Failed to segment key frames for {video_id}. Error: {stderr.decode('utf8')}")

def save_metadata_to_parquet(keyframe_video_locs, original_video_locs, directory):
    keyframe_video_df = pd.DataFrame(keyframe_video_locs)
    original_video_df = pd.DataFrame(original_video_locs)
    keyframe_video_df['duration'] = keyframe_video_df['duration'].astype(float)
    original_video_df['duration'] = original_video_df['duration'].astype(float)
    keyframe_video_df.to_parquet(f'{directory}/keyframe_video_requirements.parquet', index=False)
    original_video_df.to_parquet(f'{directory}/original_video_requirements.parquet', index=False)

def prepare_clip_encode(directories):
    base_directory = directories['base_directory']
    dataset_requirements = load_dataset_requirements(base_directory)
    df = pd.DataFrame(dataset_requirements)
    video_files = glob.glob(os.path.join(base_directory, directories["original_frames"], '**/*.mp4'), recursive=True)
    keyframe_video_locs, original_video_locs = collect_video_metadata(video_files, os.path.join(base_directory, directories["keyframes"]))
    save_metadata_to_parquet(keyframe_video_locs, original_video_locs, base_directory)

def run_video2dataset_with_yt_dlp(directories):
    base_directory = directories['base_directory']
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.makedirs(os.path.join(base_directory, directories["original_frames"]), exist_ok=True)
    url_list = os.path.join(base_directory, 'dataset_requirements.parquet')
    print(f"Reading URLs from: {url_list}")
    df = pd.read_parquet(url_list)
    for idx, row in df.iterrows():
        print(f"Processing video {idx+1}: {row['url']}")
        command = [
            'video2dataset',
            '--input_format', 'parquet',
            '--url_list', url_list,
            '--encode_formats', '{"video": "mp4", "audio": "m4a"}',
            '--output_folder', os.path.join(base_directory, directories["original_frames"]),
            '--config', os.path.join(base_path, 'pipeline', 'config.yaml')]
        result = subprocess.run(command, capture_output=True, text=True)
        print("Return code:", result.returncode)
        print("STDOUT:", result.stdout)
        
def main():
    directories = read_config(section="directory")
    if directories['video_load'] == 'directory':
        prepare_clip_encode(directories)
    else:
        print("Downloading videos from yt")
        run_video2dataset_with_yt_dlp(directories)
        fix_codecs_in_directory(directories)
        segment_key_frames_in_directory(directories)
        prepare_clip_encode(directories)
    exit_status = 0 
    print(f"Exiting {__name__} with status {exit_status}")
    return exit_status
    
if __name__ == "__main__":
    exit_status = main()
    sys.exit(exit_status)