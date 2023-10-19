import os
import glob
import shutil

def move_and_group_files():
    # Define source directories
    src_dirs = {
        'originalvideos': './datasets/originalvideos',
        'keyframevideos': './datasets/keyframes',
        'keyframeembeddings': './datasets/keyframeembeddings',
        'keyframe_clips': './output/keyframe_clips',
        'keyframes': './output/keyframes',
    }
    src_dirs['keyframeembeddings_avg'] = './keyframe_clip_embeddings/keyframe_clip_segment_averages'
    # Define destination directory
    dest_dir = './completedatasets'
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    # Initialize a dictionary to keep track of integer suffixes
    integer_suffixes = {}
    # Iterate through source directories to collect integer suffixes
    for category, src_directory in src_dirs.items():
        files = glob.glob(f"{src_directory}/*")
        for file in files:
            integer_suffix = os.path.basename(file).split('.')[0]
            # Skip unwanted folders or files
            if integer_suffix.endswith("_stats"):
                continue
            if integer_suffix not in integer_suffixes:
                integer_suffixes[integer_suffix] = []
            integer_suffixes[integer_suffix].append((category, file))
    
    # Move and group files based on integer suffixes
    for integer_suffix, file_tuples in integer_suffixes.items():
        integer_dest_dir = os.path.join(dest_dir, integer_suffix)
        os.makedirs(integer_dest_dir, exist_ok=True)
        for category, file in file_tuples:
            category_dest_dir = os.path.join(integer_dest_dir, category)
            os.makedirs(category_dest_dir, exist_ok=True)
            new_file = os.path.join(category_dest_dir, os.path.basename(file))
            shutil.move(file, new_file)
            print(f"Moved {file} to {new_file}")
    # Move completed datasets
    move_completed_datasets()

def move_completed_datasets():
    src_folder = './keyframe_clip_embeddings'
    dest_folder = './completedatasets'
    # Create destination directory if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    for sub_folder in glob.glob(os.path.join(src_folder, '*')):
        sub_folder_name = os.path.basename(sub_folder)
        dest_sub_folder = os.path.join(dest_folder, sub_folder_name, 'keyframe_clip_segment_averages')
        # Create the parent integer folder if it doesn't exist
        os.makedirs(os.path.join(dest_folder, sub_folder_name), exist_ok=True)
        # Move the subfolder
        shutil.move(sub_folder, dest_sub_folder)
        print(f"Moved {sub_folder} to {dest_sub_folder}")

if __name__ == "__main__":
    move_and_group_files()
