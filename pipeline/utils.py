import os
import glob
import shutil
import os
import numpy as np
import glob
import re
from pipeline import read_config
import os
import json
import shutil
from pydub import AudioSegment
import os
import json
import shutil
from pydub import AudioSegment

directories = read_config(section="directory")
evaluations = read_config(section="evaluations")

def convert_audio_files(base_path, output_format="mp3"):
    for n in os.listdir(base_path):
        audio_clip_output_dir = os.path.join(base_path, n, 'keyframe_audio_clips', 'whisper_audio_segments')
        if not os.path.exists(audio_clip_output_dir):
            os.makedirs(audio_clip_output_dir)
        for subdir, dirs, files in os.walk(audio_clip_output_dir):
            for filename in files:
                file_path = os.path.join(subdir, filename)
                segment_info = filename.split('_')
                if filename.endswith(".flac") and len(segment_info) > 1:
                    segment_idx = segment_info[1].split('.')[0]
                    output_filename = f"keyframe_{segment_idx}.{output_format}"
                    output_path = os.path.join(audio_clip_output_dir, output_filename)
                    if os.path.exists(output_path):
                        print(f"File {output_path} already exists. Overwriting.")
                    audio = AudioSegment.from_file(file_path, format="flac")
                    audio.export(output_path, format=output_format)
                    print(f"Converted {file_path} to {output_path}")
                    os.remove(file_path)
                    print(f"Removed {file_path}")
                if filename.endswith(".json"):
                    new_json_path = os.path.join(audio_clip_output_dir, filename)
                    if file_path != new_json_path:
                        shutil.copy(file_path, new_json_path)
                        print(f"Copied {file_path} to {new_json_path}")
                    else:
                        print(f"File {new_json_path} is the same as the source. Skipping copy.")
                    with open(new_json_path, 'r', encoding='utf-8') as json_file:
                        try:
                            segments_data = json.load(json_file)
                        except json.JSONDecodeError:
                            print(f"Error reading JSON data from {new_json_path}")
                            continue
                        if filename == ("outputs.json"):
                            for segment_data in segments_data:
                                if isinstance(segment_data, dict) and "segment_idx" in segment_data:
                                    segment_idx = segment_data["segment_idx"]
                                    text_filename = f"keyframe_{segment_idx}.txt"
                                    text_path = os.path.join(audio_clip_output_dir, text_filename)
                                    with open(text_path, 'w', encoding='utf-8') as text_file: 
                                        text_file.write(segment_data.get("text", ""))
                                    print(f"Created text file for segment {segment_idx}")

def move_and_remove_subdirectory(audio_clip_output_dir):
    for subdir in os.listdir(audio_clip_output_dir):
        subdir_path = os.path.join(audio_clip_output_dir, subdir)
        if subdir.isdigit() and os.path.isdir(subdir_path) or subdir == 'full_whisper_segments' and os.path.isdir(subdir_path):
            try:
                shutil.rmtree(subdir_path)
            except Exception as e:
                print(f"Error removing {subdir_path}: {e}")
                
def convert_types():
    base_directory = evaluations['completedatasets']
    for n in os.listdir(base_directory):
        audio_clip_output_dir = os.path.join(base_directory, n ,'keyframe_audio_clips', 'whisper_audio_segments')
        convert_audio_files(base_directory)  
        move_and_remove_subdirectory(audio_clip_output_dir)

def rename_and_move_files(src_directory, dest_directory, regex_pattern=None):
    files = glob.glob(f"{src_directory}/*")
    if regex_pattern: 
        sorted_files = sorted(files, key=lambda x: int(re.search(regex_pattern, os.path.basename(x)).group(1)) if re.search(regex_pattern, os.path.basename(x)) else 0)
    else:
        sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    new_integer_values = {}
    counter = 1
    for old_file in sorted_files:
        basename = os.path.basename(old_file)
        if regex_pattern and re.search(regex_pattern, basename):
            old_integer = re.search(regex_pattern, basename).group(1)
        else:
            old_integer = basename.split('.')[0]
        if old_integer not in new_integer_values:
            new_integer_values[old_integer] = counter
            counter += 1
        new_integer = new_integer_values[old_integer]
        extension = os.path.splitext(old_file)[1][1:]
        new_file = os.path.join(dest_directory, f"{new_integer}.{extension}")
        shutil.move(old_file, new_file)
        print(f"Moved and renamed {old_file} to {new_file}")

def rename_and_move():
    base_directory = directories['base_directory']
    original_frames = os.path.join(base_directory, directories['original_frames'])
    keyframes = os.path.join(base_directory,directories['keyframes'])
    embeddings = os.path.join(base_directory, directories['embeddings'])
    rename_and_move_files(os.path.join(original_frames, '00000'), original_frames)
    rename_and_move_files(keyframes, keyframes, regex_pattern=r'(\d+)_key_frames')
    rename_and_move_files(embeddings, embeddings)

def move_or_copy(src_path, dest_dir, file_extension, video_id, copy=False):
    for file in glob.glob(f"{src_path}/**/*{file_extension}", recursive=True):
        dest_file_name = f"{video_id}_{os.path.basename(file)}"
        dest_path = os.path.join(dest_dir, dest_file_name)
        if os.path.exists(dest_path):
            print(f"File {dest_path} already exists. Skipping.")
            continue
        if copy:
            shutil.copy(file, dest_path)
        else:
            shutil.move(file, dest_path)
        print(f"{'Copied' if copy else 'Moved'} {file} to {dest_path}")

def move_or_copy_files():
    directories = read_config(section="directory")
    evaluations = read_config(section="evaluations")
    base_directory = evaluations['completedatasets']
    keyframes_segments_dir = os.path.join(base_directory, "keyframes")
    video_segments_dir = os.path.join(base_directory, "video_segments")

    os.makedirs(keyframes_segments_dir, exist_ok=True)
    os.makedirs(video_segments_dir, exist_ok=True)

    for video_id_dir in glob.glob(f"{base_directory}/*"):
        video_id = os.path.basename(video_id_dir)
        if os.path.isdir(video_id_dir) and video_id_dir not in [keyframes_segments_dir, video_segments_dir]:
            # Define source directories
            original_videos_dir = os.path.join(video_id_dir, 'originalvideos')
            embeddings_dir = os.path.join(video_id_dir, 'keyframe_embeddings')
            keyframes_dir = os.path.join(video_id_dir, 'keyframes')
            keyframe_clips_dir = os.path.join(video_id_dir, 'keyframe_clips')
            keyframe_clip_embeddings_dir = os.path.join(video_id_dir, 'keyframe_clip_embeddings')
            keyframe_audio_clips_dir = os.path.join(video_id_dir, 'keyframe_audio_clips')

            # Process files for keyframes segments
            move_or_copy(original_videos_dir, keyframes_segments_dir, '.json', video_id, copy=True)
            move_or_copy(keyframes_dir, keyframes_segments_dir, '.npy', video_id)
            move_or_copy(keyframes_dir, keyframes_segments_dir, '.png', video_id)
            move_or_copy(keyframes_dir, keyframes_segments_dir, '.json', video_id)
            move_or_copy(embeddings_dir, keyframes_segments_dir, '.npy', video_id)
            move_or_copy(embeddings_dir, keyframes_segments_dir, '.json', video_id)

            # Process files for video segments
            move_or_copy(keyframe_clips_dir, video_segments_dir, '.mp4', video_id)
            move_or_copy(keyframe_clip_embeddings_dir, video_segments_dir, '.npy', video_id)
            move_or_copy(keyframe_clip_embeddings_dir, video_segments_dir, '.json', video_id)
            # Process files for audio segments into video dir
            move_or_copy(keyframe_audio_clips_dir, video_segments_dir, '.txt', video_id)
            move_or_copy(keyframe_audio_clips_dir, video_segments_dir, '.json', video_id)
            move_or_copy(keyframe_audio_clips_dir, video_segments_dir, '.mp3', video_id)
            shutil.rmtree(video_id_dir)
            print(f"Removed directory: {video_id_dir}")
            
def create_parquet_from_videos(video_dir, parquet_file):
    data = []
    for index, file_name in enumerate(os.listdir(video_dir)):
        if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, file_name)
            label = f'video{index + 1}'
            data.append({'path': video_path, 'label': label})
    df = pd.DataFrame(data)
    df.to_parquet(parquet_file)
    print(f"Parquet file saved to {parquet_file}")

def aggregate_and_save_npy(embedding_dir, output_file, suffixes):
    for suffix in suffixes:
        aggregated_embeddings = []
        for file_name in os.listdir(embedding_dir):
            if file_name.endswith(suffix):
                file_path = os.path.join(embedding_dir, file_name)
                embedding = np.load(file_path)
                aggregated_embeddings.append(embedding)
        if aggregated_embeddings:
            aggregated_embeddings = np.concatenate(aggregated_embeddings)
            modified_output_file = output_file.replace('.npy', f'_{suffix}')
            np.save(modified_output_file, aggregated_embeddings)
            print(f"Aggregated embeddings saved to {modified_output_file}")
        else:
            print(f"No embeddings found to aggregate for suffix {suffix}.")