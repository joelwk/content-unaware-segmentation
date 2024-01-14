from pipeline import read_config,install_local_package
import subprocess
import argparse
import os
from contextlib import contextmanager

directories = read_config(section="directory")
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
clipencode_abs_path = os.path.join(base_path,'pipeline', 'clip-video-encode')

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

def clip_encode():
    clipencode_abs_path = os.path.join(base_path,'pipeline', 'clip-video-encode')
    embeddings = os.path.join(directories['base_directory'], directories['embeddings'])
    with change_directory(clipencode_abs_path):
        from clip_video_encode import clip_video_encode
    clip_video_encode(
            f'{directories["base_directory"]}/keyframe_video_requirements.parquet',
                embeddings,
                frame_workers=int(directories['frame_workers']),
                take_every_nth=int(directories['take_every_nth']),
                metadata_columns=['videoLoc', 'videoID', 'duration'])

def main():
    clip_encode()
    return 0
if __name__ == "__main__":
    main()