import os
import subprocess
import argparse
import shutil
from contextlib import contextmanager
def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

def clone_repository(git_url, target_dir):
    repo_name = git_url.split("/")[-1].replace(".git", "")
    full_path = os.path.join(target_dir, repo_name)
    if not os.path.exists(full_path):
        subprocess.run(["git", "clone", git_url, full_path])
    return full_path

@contextmanager
def change_directory(destination):
    original_path = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(original_path)

def install_local_package(directory):
    original_directory = os.getcwd()
    if os.path.exists(directory):
        os.chdir(directory)
        subprocess.run(["pip", "install", "-e", "."])
        os.chdir(original_directory)
    else:
        print(f"Directory {directory} does not exist. Skipping package installation.")


def create_directories(config):
    for key, path in config.items():
        if not path.endswith(('.parquet', '.yaml')):
            os.makedirs(path, exist_ok=True)

def generate_config(base_directory):
    return {
        "directory": base_directory,
        "original_videos": f"{base_directory}/originalvideos",
        "keyframe_videos": f"{base_directory}/keyframes",
        "embedding_output": f"{base_directory}/originalembeddings",
        "keyframe_embedding_output": f"{base_directory}/keyframeembeddings",
        "keyframe_parquet": f"{base_directory}/keyframe_video_requirements.parquet",
        "config_yaml": f"{base_directory}/config.yaml"
    }

def safe_delete(directory):
    try:
        shutil.rmtree(directory)
        return True
    except Exception as e:
        print(f"An error occurred while deleting the directory: {e}")
        return False

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

if __name__ == "__main__":
    args = parse_args()
    config = {
        "local": generate_config("./datasets")
    }
    selected_config = config[args.mode]
    create_directories(selected_config)

    # Clone and modify video2dataset
    video2dataset_path = clone_repository("https://github.com/iejMac/video2dataset.git", "./repos")
    target_packages = {
        "pandas": ">=1.1.5,<2",
        "pyarrow": ">=6.0.1,<8",
        "imageio-ffmpeg": ">=0.4.0,<1",
    }

    modify_requirements_txt(f"{video2dataset_path}/requirements.txt", target_packages)
    
    # Add the additional package
    with open(f"{video2dataset_path}/requirements.txt", "a") as f:
        f.write("imagehash>=4.3.1\n")

    install_local_package(video2dataset_path)

    clip_video_encode_path = clone_repository("https://github.com/iejMac/clip-video-encode.git", "./repos")
    # Use it before renaming
    if safe_delete(clip_video_encode_path):
        clip_video_encode_path = clone_repository("https://github.com/iejMac/clip-video-encode.git", "./repos")
        new_path = "./repos/clipencode"
        shutil.move(clip_video_encode_path, new_path)
        with open("clipencode_path.txt", "w") as f:
            f.write(new_path)  # Write the new path to the file
