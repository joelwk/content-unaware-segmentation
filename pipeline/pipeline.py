import os
import logging
import json
import pandas as pd
import glob
import re
import sys
import subprocess
import argparse
from contextlib import contextmanager
import configparser
import shutil

def read_config(section="directory"):
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path=f'{base_path}/pipeline/config.ini'
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config.sections():
        print(f"Section '{section}' not found in configuration.")
        raise KeyError(f"Section not found: {section}")
    return {key: config[section][key] for key in config[section]}

def string_to_bool(string_value):
    return string_value.lower() in ['true', '1', 't', 'y', 'yes', 'on']

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

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()
    
def generate_config(base_directory):
    return {
        "directory": base_directory,
        "original_videos": f"{base_directory}/originalvideos",
        "keyframe_videos": f"{base_directory}/keyframes",
        "keyframe_embedding_output": f"{base_directory}/keyframeembeddings",
        "config_yaml": f"{base_directory}/config.yaml"}

def is_directory_empty(directory): 
    return not os.listdir(directory)

def delete_associated_files(video_id, params):
    try:
        video_id_str = str(video_id)
        for directory_key in ['originalframes', 'keyframes', 'embeddings', 'keyframe_outputs', 'keyframe_audio_clip_output']:
            directory = params.get(directory_key)
            if directory and os.path.exists(directory):
                # Match files that exactly start with the video_id followed by non-numeric characters
                pattern = f"{directory}/{video_id_str}[^0-9]*"
                for file in glob.glob(pattern):
                    if os.path.isfile(file):
                        os.remove(file)
                        logging.warning(f"Deleted file {file} associated with video {video_id}.")
                    elif os.path.isdir(file):
                        shutil.rmtree(file)
                        logging.warning(f"Deleted directory {file} associated with video {video_id}.")
    except Exception as e:
        print(f"Error in deleting files for video ID {video_id}: {e}")

def create_directories(config):
    for key, path in config.items():
        if not path.endswith(('.parquet', '.yaml')):
            os.makedirs(path, exist_ok=True)

def get_local_package_dependencies(directory):
    setup_file = os.path.join(directory, 'setup.py')
    if not os.path.exists(setup_file):
        print(f"Setup file not found in {directory}")
        return []
    dependencies = []
    with open(setup_file, 'r') as file:
        for line in file:
            if 'install_requires' in line:
                deps = re.findall(r"'([^']*)'", line)
                dependencies.extend(deps)
    return dependencies

def clone_repository(git_url, target_dir):
    repo_name = git_url.split("/")[-1].replace(".git", "")
    full_path = os.path.join(target_dir, repo_name)
    if not os.path.exists(full_path):
        result = subprocess.run(["git", "clone", git_url, full_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error cloning repository: {result.stderr}")
            return 1
    return full_path

def install_local_package(directory):
    with change_directory(directory):
        result = subprocess.run(["pip", "install", "-e", "."], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error installing local package: {result.stderr}")
            return 1

def modify_requirements_txt(file_path, target_packages):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        for line in lines:
            modified = False
            for package, new_version in target_packages.items():
                if line.startswith(package):
                    f.write(f"{package}{new_version}\n")
                    modified = True
                    break
            if not modified:
                f.write(line)

def install_requirements(directory):
    req_file = os.path.join(directory, 'requirements.txt')
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-r", req_file], check=True)

def prepare_dataset_requirements(directories, external_parquet_path):
    try:
        if external_parquet_path is not None:
            shutil.copy(external_parquet_path, f"{directories}/dataset_requirements.parquet")
            print(f"Copied external Parquet file to {directories}")
        else:
            dataset_requirements = {
                "data": [
                    {"url": "www.youtube.com/watch?v=iqrpwARx26E", "caption": "Elon Musk on politics: I would not vote for a pro-censorship candidate"},
                    {"url": "www.youtube.com/watch?v=JKNyNJT4wzg", "caption": "Viktor Orban Blocks EU's €50 Billion Ukraine Aid Package"},
                    {"url": "www.youtube.com/watch?v=YEUclZdj_Sc", "caption": "Why next-token prediction is enough for AGI"},]}
            df = pd.DataFrame(dataset_requirements['data'])
            parquet_file_path = f"{directories}/dataset_requirements.parquet"
            df.to_parquet(parquet_file_path, index=False)
            print(f"Saved Parquet file at {parquet_file_path}")
    except Exception as e:
        print(f"Error while saving Parquet file: {e}")

def main():
    directories = read_config(section="directory")
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    external_parquet = directories.get("external_parquet", None)
    try:
        args = parse_args()
        config = {"local": generate_config(directories['base_directory'])}
        selected_config = config[args.mode]
        create_directories(selected_config)
        clipvideoencode_repo_url = "https://github.com/iejMac/clip-video-encode.git"
        video2dataset_repo_url = "https://github.com/iejMac/video2dataset.git"
        clipvideoencode_requirements_directory = clone_repository(clipvideoencode_repo_url, os.path.join(base_path,'pipeline'))
        video2dataset_requirements_directory = clone_repository(video2dataset_repo_url, os.path.join(base_path,'pipeline'))
        requirements_general =  os.path.join(base_path, 'pipeline')
        install_requirements(requirements_general)
        if directories['video_load'] == 'download':
            # If videos need to be downloaded, install both video2dataset & clip-video-encode packages
            install_local_package(video2dataset_requirements_directory)
            if external_parquet == "None":
                external_parquet = None
            prepare_dataset_requirements(directories["base_directory"], external_parquet)
        else:
            # If videos are already downloaded, install clip-video-encode package to start segmentation
            install_requirements(clipvideoencode_requirements_directory)
    except Exception as e:
        print(f"An exception occurred during pip install: {e}")
        return 1
if __name__ == "__main__":
    sys.exit(main())