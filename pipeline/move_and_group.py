import os
import glob
import shutil

def move_and_group_files():
    # Define source directories
    src_dirs = {
        'originalvideos': './datasets/originalvideos',
        'keyframevideos': './datasets/keyframes',
        'keyframeembeddings': './datasets/keyframeembeddings',
        'keyframe_clips': './output/keyframe_clip',
        'keyframes': './output/keyframes',
        'keyframe_clip_embeddings': './keyframe_clip_embeddings/'
    }
    
    # Define destination directory
    dest_dir = './completedatasets'
    os.makedirs(dest_dir, exist_ok=True)
    
    # Initialize a dictionary to keep track of integer suffixes
    integer_suffixes = {}
    
    for category, src_directory in src_dirs.items():
        for file_path in glob.glob(f"{src_directory}/*"):
            basename = os.path.basename(file_path)
            
            # Skip stats files
            if basename.endswith("_stats"):
                continue
            
            # Extract integer suffix
            integer_suffix = basename.split('.')[0]
            
            if category in ['keyframes', 'keyframe_clips', 'keyframe_clip_embeddings']:
                # For these categories, go one level deeper
                for nested_file in glob.glob(f"{file_path}/*"):
                    if integer_suffix not in integer_suffixes:
                        integer_suffixes[integer_suffix] = []
                    integer_suffixes[integer_suffix].append((category, nested_file))
            else:
                if integer_suffix not in integer_suffixes:
                    integer_suffixes[integer_suffix] = []
                integer_suffixes[integer_suffix].append((category, file_path))
                
    for integer_suffix, file_tuples in integer_suffixes.items():
        integer_dest_dir = os.path.join(dest_dir, integer_suffix)
        os.makedirs(integer_dest_dir, exist_ok=True)
        
        for category, file_path in file_tuples:
            category_dest_dir = os.path.join(integer_dest_dir, category)
            os.makedirs(category_dest_dir, exist_ok=True)
            
            new_file_path = os.path.join(category_dest_dir, os.path.basename(file_path))
            shutil.move(file_path, new_file_path)
            print(f"Moved {file_path} to {new_file_path}")

if __name__ == "__main__":
    move_and_group_files()
