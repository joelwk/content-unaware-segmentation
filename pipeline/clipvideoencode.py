from pipeline import read_config
import subprocess
import argparse
import os

def install_clip_video_encode():
    subprocess.run(["pip", "install", "clip-video-encode"], check=True)

def clip_encode(selected_config):

    # Dynamically set the base path
    if 'content' in os.getcwd():
        base_path = '/content'  # Google Colab
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # Local environment

    # Construct the absolute path for clip-video-encode
    clipencode_abs_path = os.path.join(base_path, selected_config['main_repo'], 'clip-video-encode')


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
    if 'content' in os.getcwd():
        base_path = '/content'
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Install clip-video-encode
    install_clip_video_encode()
    # Run the clip encoding
    clip_encode(directories)

    return 0

if __name__ == "__main__":
    main()
