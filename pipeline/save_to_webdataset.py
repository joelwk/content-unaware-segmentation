import os
import glob
from webdataset import ShardWriter
import numpy as np
import json
from io import BytesIO
import ast
from pipeline import read_config

def datasets_to_webdataset_segmentation(root_folder, output_folder, shard_size=1e9):
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
                extension = os.path.splitext(item)[-1][1:] 
                if extension == 'npy':
                    array = np.load(item_path)
                    assert isinstance(array, np.ndarray)
                    sample[category] = array
                elif extension == 'json':
                    with open(item_path, 'r') as f:
                        json_data = json.load(f)
                        assert isinstance(json_data, (list, dict))
                        sample[category] = json_data
                else:
                    with open(item_path, 'rb') as f:
                        buffer = BytesIO(f.read())
                        assert isinstance(buffer, BytesIO)
                        sample[category] = buffer.getvalue()
    with ShardWriter(pattern, maxsize=shard_size) as sink:
        for i, dataset_folder in enumerate(dataset_folders):
            sample = {}
            sample['__key__'] = os.path.basename(dataset_folder)
            assert isinstance(sample['__key__'], str)
            recursive_add_files(dataset_folder, sample, sample['__key__'])
            sink.write(sample)

def datasets_to_webdataset_evaluations(root_folder, output_folder, shard_size=1e9):
    dataset_folders = sorted(glob.glob(f"{root_folder}/*"))
    pattern = os.path.join(output_folder, "completed_evaluations-%06d.tar")
    def recursive_add_files(folder_path, sample, parent_key):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                new_key = f"{parent_key}/{item}"
                recursive_add_files(item_path, sample, new_key)
            else:
                category = f"{parent_key}/{item}"
                extension = os.path.splitext(item)[-1][1:] 
                if extension == 'npy':
                    array = np.load(item_path)
                    assert isinstance(array, np.ndarray)
                    sample[category] = array
                elif extension == 'json':
                    with open(item_path, 'r') as f:
                        json_data = json.load(f)
                        assert isinstance(json_data, (list, dict))
                        sample[category] = json_data
                else:
                    with open(item_path, 'rb') as f:
                        buffer = BytesIO(f.read())
                        assert isinstance(buffer, BytesIO)
                        sample[category] = buffer.getvalue()
    with ShardWriter(pattern, maxsize=shard_size) as sink:
        for i, dataset_folder in enumerate(dataset_folders):
            sample = {}
            sample['__key__'] = os.path.basename(dataset_folder)
            assert isinstance(sample['__key__'], str)
            recursive_add_files(dataset_folder, sample, sample['__key__'])
            sink.write(sample)

def package_datasets_to_webdataset_evaluations():
    directories = read_config(section="directory")
    evaluations = read_config(section="evaluations")
    root_folder = evaluations['outputs']
    output_folder = directories['video_wds_output']
    os.makedirs(output_folder, exist_ok=True)
    datasets_to_webdataset_evaluations(root_folder, output_folder)

def package_datasets_to_webdataset_segmentation():
    directories = read_config(section="directory")
    evaluations = read_config(section="evaluations")
    root_folder = evaluations['completedatasets']
    output_folder = directories['video_wds_output']
    os.makedirs(output_folder, exist_ok=True)
    datasets_to_webdataset_segmentation(root_folder, output_folder)
