import os
import json
import sys
import subprocess
import argparse
from contextlib import contextmanager

def modify_hook_file(hook_file_path):
    if hook_file_path:
        try:
            with open(hook_file_path, 'r') as file:
                lines = file.readlines()
            with open(hook_file_path, 'w') as file:
                for line in lines:
                    if "from training.data import get_audio_features" in line:
                        line = line.replace("from training.data import get_audio_features", 
                                            "from laion_clap.training.data import get_audio_features")
                    elif "from training.data import int16_to_float32, float32_to_int16" in line:
                        line = line.replace("from training.data import int16_to_float32, float32_to_int16", 
                                            "from laion_clap.training.data import int16_to_float32, float32_to_int16")
                    file.write(line)
            print(f"Modified hook.py at {hook_file_path}")
        except Exception as e:
            print(f"Error modifying hook.py: {e}")
            
def install_requirements():
    try:
        print("Installing required packages...")
        subprocess.run(["pip", "install", "yt-dlp"])
        subprocess.run(["pip", "install", "laion_clap"])
        subprocess.run(["pip", "install", "open_clip_torch"])
        subprocess.run(["pip", "install", "scikit-learn==1.3.0"])
        subprocess.run(["pip", "install", "pydub"])
        subprocess.run(["pip", "install", "demucs"])
    except:
        print("Failed to install required packages.")
        return 1

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Configuration')
    parser.add_argument('--mode', type=str, default='local', help='Execution mode: local or cloud')
    return parser.parse_args()

def generate_config(base_directory):
    return {
        "evaluations": base_directory,
        "labels": f"{base_directory}/embeddings",
        "image_audio_pairs": f"{base_directory}/image_audio_pairs",
        "paired_evaluations": f"{base_directory}/paired_evaluations",
        "image_evaluations": f"{base_directory}/image_evaluations",
    }

def create_directories(config):
    for key, path in config.items():
        if not path.endswith(('.parquet', '.yaml')):
            os.makedirs(path, exist_ok=True)

def main():
    try:
        args = parse_args()
        config = {"local": generate_config("./evaluations")}
        selected_config = config[args.mode]
        create_directories(selected_config)
        install_requirements()
    except Exception as e:
        print(f"An exception occurred: {e}")
        return 1
    return 0 
    
if __name__ == "__main__":
    sys.exit(main()) 