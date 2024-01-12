from pipeline import read_config
import subprocess
import argparse
import os
from contextlib import contextmanager

def install_clip_video_encode():
    subprocess.run(["pip", "install", "clip-video-encode"], check=True)

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

def clip_encode(selected_config):
  base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  clipencode_abs_path = os.path.join(base_path,'pipeline', 'clip-video-encode')
  with change_directory(clipencode_abs_path):
      install_clip_video_encode()
      from clip_video_encode import clip_video_encode
  clip_video_encode(
          f'{selected_config["base_directory"]}/keyframe_video_requirements.parquet',
            selected_config["embeddings"],
            frame_workers=int(selected_config['frame_workers']),
            take_every_nth=int(selected_config['take_every_nth']),
            metadata_columns=['videoLoc', 'videoID', 'duration']
        )

def main():
    directories = read_config(section="directory")
    clip_encode(directories)
    return 0

if __name__ == "__main__":
    main()
