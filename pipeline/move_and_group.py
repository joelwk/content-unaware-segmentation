import os
import glob
import shutil
from pipeline import read_config, is_directory_empty

def move_and_group_files(directories):
    # Define source directories for various categories of files
    base_directory = directories['base_directory']
    src_dirs = {
        'originalvideos': os.path.join(base_directory, directories['original_frames']),
        'keyframes': os.path.join(base_directory, directories['keyframes']),
        'keyframe_embeddings': os.path.join(base_directory, directories['embeddings']),
        'keyframe_clips': os.path.join(base_directory, directories['keyframe_clip_output']),
        'keyframe_clip_embeddings': os.path.join(base_directory, directories['keyframe_clip_embeddings_output']),
        'keyframe_audio_clips': os.path.join(base_directory, directories['keyframe_audio_clip_output'])
    }
    dest_dir = './completedatasets'
    os.makedirs(dest_dir, exist_ok=True)
    integer_suffixes = {}
    invalid_suffixes = set()  # To track video IDs with missing/empty files
    for category, src_directory in src_dirs.items():
        for file_path in glob.glob(f"{src_directory}/*"):
            basename = os.path.basename(file_path)
            if basename.endswith("_stats"):
                continue
            integer_suffix = basename.split('.')[0]
            if category in ['keyframes','keyframe_audio_clips']:
                for nested_file in glob.glob(f"{file_path}/*"):
                    if os.path.getsize(nested_file) == 0 or is_directory_empty(file_path):
                        invalid_suffixes.add(integer_suffix)
                    else:
                        integer_suffixes.setdefault(integer_suffix, []).append((category, nested_file))
            else:
                if os.path.getsize(file_path) == 0:
                    invalid_suffixes.add(integer_suffix)
                else:
                    integer_suffixes.setdefault(integer_suffix, []).append((category, file_path))
    for integer_suffix, file_tuples in integer_suffixes.items():
        if integer_suffix in invalid_suffixes:
            print(f"Deleting files for video ID {integer_suffix} due to missing or empty files.")
            for _, file_path in file_tuples:
                os.remove(file_path)
        else:
            integer_dest_dir = os.path.join(dest_dir, integer_suffix)
            os.makedirs(integer_dest_dir, exist_ok=True)
            for category, file_path in file_tuples:
                category_dest_dir = os.path.join(integer_dest_dir, category)
                os.makedirs(category_dest_dir, exist_ok=True)
                new_file_path = os.path.join(category_dest_dir, os.path.basename(file_path))
                shutil.move(file_path, new_file_path)
                print(f"Moved {file_path} to {new_file_path}")

def cleanup_unwanted_dirs(directory, unwanted_dirs=None):
    if unwanted_dirs:
        for unwanted_dir in unwanted_dirs:
            path_to_remove = os.path.join(directory, unwanted_dir)
            if os.path.exists(path_to_remove):
                shutil.rmtree(path_to_remove)
                print(f"Removed {path_to_remove}")
    else:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed entire directory: {directory}")

def main():
    directories = read_config()
    evaluations = read_config(section="evaluations")
    move_and_group_files(directories)
    cleanup_unwanted_dirs(evaluations['completedatasets'], ['00000_stats', '00000'])
    cleanup_unwanted_dirs(directories['base_directory'])  
if __name__ == "__main__":
    main()