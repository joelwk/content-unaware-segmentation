import os
import pandas as pd
import json
import glob
import configparser
import subprocess
import argparse
import shutil
import ffmpeg
from pipeline import *

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

args = parse_args()
config = {
    "local": generate_config("./datasets")}
selected_config = config[args.mode]

def main():

    def prepare_dataset_requirements(directory, external_parquet_path=None):
        if external_parquet_path:
            # If an external Parquet file is provided, copy it to the directory
            shutil.copy(external_parquet_path, f"{directory}/dataset_requirements.parquet")
        else:
            # Otherwise, create a new Parquet file from the default JSON data
            dataset_requirements = {
                "data": [
                    {"url": "www.youtube.com/watch?v=nXBoOam5xJs", "caption": "The Deadly Portuguese Man O' War"},
                ]
            }
            os.makedirs(directory, exist_ok=True)
            df = pd.DataFrame(dataset_requirements['data'])
            df.to_parquet(f"{directory}/dataset_requirements.parquet", index=False)

    def load_dataset_requirements(directory):
        # Read from the Parquet file instead of the JSON file
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
            print(f"Loaded metadata for {video_id}: {metadata}")
            duration = metadata['video_metadata']['streams'][0]['duration']
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
    def fix_codecs_in_directory(directory):
        video_files = glob.glob(f"{directory}/**/*.mp4", recursive=True)
        for video_file in video_files:
            video_id = os.path.basename(video_file).split('.')[0]
            input_file_path = video_file 
            output_file_path = os.path.join(directory, f"fixed_{video_id}.mp4")
            try:
                ffmpeg.input(input_file_path).output(output_file_path, vcodec='libx264', strict='-2', loglevel="quiet").overwrite_output().run(capture_stdout=True, capture_stderr=True)
                print(f"Successfully re-encoded {video_file}")
                os.remove(input_file_path)
                os.rename(output_file_path, input_file_path)
            except ffmpeg.Error as e:
                print(f"Failed to re-encode {video_file}. Error: {e.stderr.decode('utf8')}")
                
    def segment_key_frames_in_directory(directory, output_directory):
        video_files = glob.glob(f"{directory}/**/*.mp4", recursive=True)
        for video_file in video_files:
            video_id = os.path.basename(video_file).split('.')[0]
            input_file = video_file
            output_file = os.path.join(output_directory, f"{video_id}_key_frames.mp4")
            print(f"Segmenting key frames for {video_id}...")
            command = f'ffmpeg -y -loglevel error -discard nokey -i {input_file} -c:s copy -c copy -copyts {output_file}'
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print(f"Successfully segmented key frames for {video_id}.")
            else:
                print(f"Failed to segment key frames for {video_id}. Error: {stderr.decode('utf8')}")

    def save_metadata_to_parquet(keyframe_video_locs, original_video_locs, directory):
        keyframe_video_df = pd.DataFrame(keyframe_video_locs)
        original_video_df = pd.DataFrame(original_video_locs)
        keyframe_video_df.to_parquet(f'{directory}/keyframe_video_requirements.parquet', index=False)
        original_video_df.to_parquet(f'{directory}/original_video_requirements.parquet', index=False)

    def prepare_clip_encode(directory, output):
        dataset_requirements = load_dataset_requirements(directory)
        df = pd.DataFrame(dataset_requirements)

        video_files = glob.glob(f"{selected_config['original_videos']}/**/*.mp4", recursive=True)
        keyframe_video_locs, original_video_locs = collect_video_metadata(video_files, output)
        save_metadata_to_parquet(keyframe_video_locs, original_video_locs, selected_config["directory"])

    def run_video2dataset_with_yt_dlp(directory, output):
        os.makedirs(output, exist_ok=True)
        url_list = f'{directory}/dataset_requirements.parquet'
        print(f"Reading URLs from: {url_list}")
        df = pd.read_parquet(url_list)
        for idx, row in df.iterrows():
            print(f"Processing video {idx+1}: {row['url']}")
            command = [
                'video2dataset',
                '--input_format', 'parquet',
                '--url_list', url_list,
                '--encode_formats', '{"video": "mp4", "audio": "m4a"}',
                '--output_folder', output,
                '--config', '/content/config.yaml'
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            print("Return code:", result.returncode)
            print("STDOUT:", result.stdout)

    external_parquet_path = "./video_urls.parquet"  # Replace with actual path or None
    prepare_dataset_requirements(selected_config["directory"], external_parquet_path = None)
    run_video2dataset_with_yt_dlp(selected_config["directory"], selected_config["original_videos"])
    fix_codecs_in_directory(selected_config["original_videos"])
    segment_key_frames_in_directory(selected_config["original_videos"], selected_config["keyframe_videos"])
    prepare_clip_encode(selected_config["directory"], selected_config["keyframe_videos"])

if __name__ == "__main__":
  main()