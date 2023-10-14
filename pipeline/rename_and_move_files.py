import os
import glob
import shutil
import re

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

if __name__ == "__main__":
    # Rename and move all files from the originalvideos/00000 directory to originalvideos/
    rename_and_move_files("./datasets/originalvideos/00000", "./datasets/originalvideos")
    # Rename all files in the keyframes directory
    rename_and_move_files("./datasets/keyframes", "./datasets/keyframes", regex_pattern=r'(\d+)_key_frames')
    # Rename all files in the keyframeembeddings directory
    rename_and_move_files("./datasets/keyframeembeddings", "./datasets/keyframeembeddings")
