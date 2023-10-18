import os
import glob
import tarfile
from webdataset import ShardWriter

def package_datasets_to_webdataset(root_folder, output_folder, shard_size=1e9):
    dataset_folders = sorted(glob.glob(f"{root_folder}/*"))
    pattern = os.path.join(output_folder, "completed_datasets-%06d.tar")

    def recursive_add_files(folder_path, sample, parent_key):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            if os.path.isdir(item_path):
                new_key = f"{parent_key}/{item}"
                recursive_add_files(item_path, sample, new_key)
            else:
                category = f"{parent_key}/{item}"
                extension = os.path.splitext(item)[-1]
                
                with open(item_path, 'rb') as f:
                    sample[f"{category}{extension}"] = f.read()

    with ShardWriter(pattern, maxsize=shard_size) as sink:
        for i, dataset_folder in enumerate(dataset_folders):
            sample = {}
            sample['__key__'] = os.path.basename(dataset_folder)
            
            recursive_add_files(dataset_folder, sample, sample['__key__'])
            
            sink.write(sample)

if __name__ == '__main__':
    root_folder = '/content/completedatasets'
    output_folder = '/content/shards'
    os.makedirs(output_folder, exist_ok=True)
    package_datasets_to_webdataset(root_folder, output_folder)
