import os
import argparse
import subprocess
import pandas as pd
from pipeline import *

def install_local_package(directory):
    original_directory = os.getcwd()
    if os.path.exists(directory):
        print(f"Changing directory to {directory}")
        os.chdir(directory)
        subprocess.run(["pip", "install", "-e", "."])
        os.chdir(original_directory)
        print(f"Changed directory back to {original_directory}")
        return True
    else:
        print(f"Directory {directory} does not exist. Skipping package installation.")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

args = parse_args()

config = {
    "local": generate_config("./datasets")}
selected_config = config[args.mode]

def clip_encode():
    clip_video_encode(
        selected_config["keyframe_parquet"],
        selected_config["keyframe_embedding_output"],
        frame_workers=25,
        take_every_nth=1,
        metadata_columns=['videoLoc', 'videoID', 'duration']
    )

def get_clipencode_path():
    try:
        with open("clipencode_path.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("clipencode_path.txt not found. Make sure to run the pipeline setup script first.")
        return None

if __name__ == "__main__":
  
    clipencode_path = get_clipencode_path()
    if clipencode_path:
        success = install_local_package(clipencode_path)
        if success:
            try:
                from repos.clipencode.clip_video_encode import clip_video_encode
                clip_encode()
            except ImportError:
                print("Could not import clip_video_encode despite successful installation.")
        else:
            print("Could not import clip_video_encode due to installation failure.")