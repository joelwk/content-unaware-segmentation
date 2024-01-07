import os
import glob
import shutil
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
    base_path = './completedatasets'
    for n in os.listdir(base_path):
        audio_clip_output_dir = os.path.join(base_path, n ,'keyframe_audio_clips', 'whisper_audio_segments')
        convert_audio_files(base_path)  
        move_and_remove_subdirectory(audio_clip_output_dir)

def rename_and_move_files(src_directory, dest_directory, regex_pattern=None):
    # Get all files in the source directory
    files = glob.glob(f"{src_directory}/*")
    # Sort the files based on the last integer in their names
    if regex_pattern:
        sorted_files = sorted(files, key=lambda x: int(re.search(regex_pattern, os.path.basename(x)).group(1)) if re.search(regex_pattern, os.path.basename(x)) else 0)
    else:
        sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
        
    # Initialize a dictionary to keep track of the new integer values
    new_integer_values = {}
    # Rename and move the files
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
        extension = os.path.splitext(old_file)[1][1:]  # Extract the file extension
        new_file = os.path.join(dest_directory, f"{new_integer}.{extension}")
        shutil.move(old_file, new_file)
        print(f"Moved and renamed {old_file} to {new_file}")

def rename_and_move():
    directories = read_config(section="directory")
    # Rename and move all files from the originalvideos/00000 directory to originalvideos/
    rename_and_move_files(f"{directories['originalframes']}/00000", directories['originalframes'])
    # Rename all files in the keyframes directory
    rename_and_move_files(directories['keyframes'], directories['keyframes'], regex_pattern=r'(\d+)_key_frames')
    # Rename all files in the keyframeembeddings directory
    rename_and_move_files(directories['embeddings'], directories['embeddings'])