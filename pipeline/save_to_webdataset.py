import os
import glob
from webdataset import ShardWriter
import numpy as np
import json
from io import BytesIO
import ast

def package_datasets_to_webdataset(root_folder, output_folder, shard_size=1e9):
    # List all folders in the root directory and sort them
    dataset_folders = sorted(glob.glob(f"{root_folder}/*"))
    
    # Create a pattern for shard file naming
    pattern = os.path.join(output_folder, "completed_datasets-%06d.tar")
    
    def recursive_add_files(folder_path, sample, parent_key):
        # Loop through each file and folder in the directory
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # If it's a directory, recurse into it
            if os.path.isdir(item_path):
                new_key = f"{parent_key}/{item}"
                recursive_add_files(item_path, sample, new_key)
            else:
                # Prepare a key to represent this data item
                category = f"{parent_key}/{item}"
                extension = os.path.splitext(item)[-1][1:]  # Strip the dot from the extension
                
                # Explicitly check types and add assert statements
                
                # Handle .npy files (NumPy arrays)
                if extension == 'npy':
                    array = np.load(item_path)
                    assert isinstance(array, np.ndarray)
                    sample[f"{category}"] = array  
                elif extension == 'json':
                    # Handle JSON files
                    with open(item_path, 'r') as f:
                        json_data = json.load(f)
                        assert isinstance(json_data, (list, dict))
                        sample[f"{category}.json"] = json_data
                else:
                    # Handle other types of files
                    with open(item_path, 'rb') as f:
                        buffer = BytesIO(f.read())  # Read file as a byte stream and store in a BytesIO buffer
                        assert isinstance(buffer, BytesIO)
                        sample[f"{category}.{extension}"] = buffer.getvalue()  # Get the byte stream from the buffer

    # Initialize ShardWriter with the naming pattern and maximum shard size
    with ShardWriter(pattern, maxsize=shard_size) as sink:
        # Loop through each dataset folder
        for i, dataset_folder in enumerate(dataset_folders):
            sample = {}
            
            # Use the folder name as a key for the entire sample
            sample['__key__'] = os.path.basename(dataset_folder)
            assert isinstance(sample['__key__'], str)
            
            # Populate the sample dictionary with data
            recursive_add_files(dataset_folder, sample, sample['__key__'])
            
            # Write the sample into a shard
            sink.write(sample)

if __name__ == '__main__':
    root_folder = './completedatasets'
    output_folder = '.'
    os.makedirs(output_folder, exist_ok=True)
    package_datasets_to_webdataset(root_folder, output_folder)
