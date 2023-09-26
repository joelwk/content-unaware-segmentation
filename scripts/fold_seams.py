import os
import json
import cv2
import subprocess
from load_data import read_config, get_all_video_ids, load_video_files, load_key_video_files, load_embedding_files, load_embedding_values, get_video_duration

def run_ffmpeg_command(start_time, end_time, input_path, output_path):
    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', input_path,
        '-c:v', 'copy', '-c:a', 'copy',
        '-y', output_path
    ]
    subprocess.run(command)

def segment_video_using_keyframes_and_embeddings(video_path, cut_output_dir, keyframe_timestamps, frame_embedding_pairs, timestamps, suffix_=None):
    video_key_frames = cv2.VideoCapture(video_path)
    writer = None
    current_keyframe = 0
    frame_rate = int(video_key_frames.get(cv2.CAP_PROP_FPS))
    start_time = 0
    segment_idx = 0
    while True:
        ret, frame = video_key_frames.read()
        if not ret:
            break
        current_time = video_key_frames.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_keyframe < len(keyframe_timestamps) - 1 and current_time >= keyframe_timestamps[current_keyframe]:
            end_time = current_time
            output_path = f"{cut_output_dir}/cut_segment_{segment_idx}_{suffix_}.mp4"
            command = [
                'ffmpeg',
                '-ss', str(start_time),      # start time
                '-to', str(end_time),        # end time
                '-i', video_path,            # input file
                '-c:v', 'copy', '-c:a', 'copy', # codec options: copy both audio and video streams
                '-y', output_path            # output file
            ]
            subprocess.run(command)
            start_time = current_time
            segment_idx += 1
            current_keyframe += 1

    # Process the final segment
    if start_time < current_time:
        output_path = f"{cut_output_dir}/cut_segment_{segment_idx}_{suffix_}.mp4"
        command = [
            'ffmpeg',
            '-ss', str(start_time),
            '-to', str(current_time),
            '-i', video_path,
            '-c:v', 'copy', '-c:a', 'copy',
            '-y', output_path
        ]
        subprocess.run(command)
    
    video_key_frames.release()


def get_segmented_frames_and_embeddings(video_files, embedding_values, total_duration):
    frame_embedding_pairs = []
    timestamps = []
    vid_cap = cv2.VideoCapture(video_files[0])
    for emb_idx, embedding in enumerate(embedding_values):
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, emb_idx)
        success, frame = vid_cap.read()
        if not success:
            break
        timestamp = (total_duration / len(embedding_values)) * emb_idx
        frame_embedding_pairs.append((frame, embedding))
        timestamps.append(timestamp)
    vid_cap.release()
    return frame_embedding_pairs, timestamps

def main(specific_videos=None):
    params = read_config(section="directory")
    video_ids = get_all_video_ids(params['originalframes']) if specific_videos is None else specific_videos
    for vid in video_ids:
        setup_for_video(vid, params)

def setup_for_video(vid, params):
    video_files = load_video_files(str(vid), params)
    key_video_files = load_key_video_files(str(vid), params)
    embedding_files = load_embedding_files(str(vid), params)
    embedding_values = load_embedding_values(embedding_files)
    total_duration = get_video_duration(video_files)
    clip_output = f"./datasets/originalvideos/cut_segments/{vid}"
    os.makedirs(clip_output, exist_ok=True)
    # Load the json with index timestamps
    json_path = f"./datasets/originalvideos/keyframes/{vid}/keyframe_data.json"
    with open(json_path, 'r') as f:
        keyframe_data = json.load(f)
    # Get the timestamp and frame index for each keyframe
    keyframe_timestamps = [frame_data['time_frame'] for frame_data in keyframe_data.values()]
    frame_embedding_pairs, timestamps = get_segmented_frames_and_embeddings(video_files, embedding_values, total_duration)
    # Process segments sourced from the key frame videos and original videos
    segment_video_using_keyframes_and_embeddings(key_video_files[0], clip_output, keyframe_timestamps, frame_embedding_pairs, timestamps,suffix_='_fromkeyvideo')
    segment_video_using_keyframes_and_embeddings(video_files[0], clip_output, keyframe_timestamps, frame_embedding_pairs, timestamps,suffix_='_fromfullvideo')


if __name__ == "__main__":
    main()
