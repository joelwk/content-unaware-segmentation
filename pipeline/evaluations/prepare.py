import configparser
import shutil
import sys
import os
import pickle
import json
import re
import glob
import subprocess
import traceback
import re
import io
from pydub import AudioSegment
from evaluations.pipeline_eval import modify_hook_file

import pandas as pd
import numpy as np
try:
    import laion_clap
    import open_clip
    import tensorflow as tf
    import torch
    from PIL import Image
except ImportError as e:
    tb = traceback.format_exc()
    match = re.search(r'File "(.*?/laion_clap/hook\.py)", line \d+, in', tb)
    if match:
        hook_file_path = match.group(1)
        print(f"Path to hook.py: {hook_file_path}")
        modify_hook_file(hook_file_path)
        import laion_clap
        import open_clip
        import tensorflow as tf
        import torch
        from PIL import Image
    else:
        print("Could not find the path to hook.py in the ImportError traceback.")

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config_path = f'{base_path}/config.ini'

def read_config(section, config_path=config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config.sections():
        raise KeyError(f"Section {section} not found in configuration file.")
    return {key: config[section][key] for key in config[section]}

evaluations = read_config('evaluations', config_path)
model_config = read_config('evaluations', config_path)
model_config = read_config('evaluations', config_path)
labels = read_config('labels', config_path)

def load_key_image_files(vid, params):
    pattern = os.path.join(params['completedatasets'], str(vid), "keyframes", "*.png")
    return iter(sorted(glob.glob(pattern)))

def load_key_audio_files(vid, params):
    pattern = os.path.join(params['completedatasets'], str(vid), "keyframe_audio_clips", "whisper_audio_segments", "*.mp3")
    return iter(sorted(glob.glob(pattern)))

def get_all_video_ids(directory):
    return iter([int(os.path.basename(f)) for f in glob.glob(os.path.join(directory, '*'))])

def tensor_to_array(tensor):
    return tensor.cpu().numpy()

def generate_embeddings(tokenizer, model_clip, prompts, file_name):
    if not os.path.exists(file_name + '.npy'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        text = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = model_clip.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = tensor_to_array(text_features) 
        np.save(file_name, text_features)
    else:
        text_features = np.load(file_name + '.npy')
    return text_features

def remove_duplicate_extension(filename):
    parts = filename.split('.')
    if len(parts) > 2 and parts[-1] == parts[-2]:
        return '.'.join(parts[:-1])
    return filename

def normalize_scores(scores):
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)
    normalized_scores = (scores - mean) / std
    return normalized_scores

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def sort_and_store_scores(probabilities, labels):
    min_length = min(len(probabilities), len(labels))
    scores = {labels[i]: float(probabilities[i]) for i in range(min_length)}
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_scores

def get_video_ids(directory):
    all_items = os.listdir(directory)
    video_ids = [item for item in all_items if os.path.isdir(os.path.join(directory, item)) and re.match(r'^\d+\.\d+$', item)]
    return video_ids

def process_keyframe_audio_pairs(faces_dir, audio_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio_dir = os.path.join(audio_dir, "whisper_audio_segments")
    text_dir_whisper = os.path.join(audio_dir, "whisper_text_segments")
    text_dir_yt = os.path.join(audio_dir.replace("whisper_audio_segments", ""), "yt_audio_segments")
    keyframe_filenames = [f for f in os.listdir(faces_dir) if f.endswith('.png')]
    for keyframe_filename in keyframe_filenames:
        segment_match = re.search(r'keyframe_(\d+)', keyframe_filename)
        if segment_match:
            segment_idx = segment_match.group(1)
            audio_filename = f"keyframe_{segment_idx}.mp3"
            text_filename_whisper = f"keyframe_{segment_idx}_transcripts.txt"
            text_filename_yt = f"keyframe_{segment_idx}_yt_transcripts.txt"
            audio_path = os.path.join(audio_dir, audio_filename)
            text_path_whisper = os.path.join(text_dir_whisper, text_filename_whisper)
            text_path_yt = os.path.join(text_dir_yt, text_filename_yt)
            image_path = os.path.join(faces_dir, keyframe_filename)
            if os.path.isfile(audio_path):
                output_audio_path = os.path.join(output_dir, audio_filename)
                shutil.copy(audio_path, output_audio_path)
                print(f"Copied {audio_path} to {output_audio_path}")
            if os.path.isfile(text_path_whisper):
                output_text_path = os.path.join(output_dir, text_filename_whisper)
                shutil.copy(text_path_whisper, output_text_path)
                print(f"Copied {text_path_whisper} to {output_text_path}")
            elif os.path.isfile(text_path_yt):
                output_text_path = os.path.join(output_dir, text_filename_yt)
                shutil.copy(text_path_yt, output_text_path)
                print(f"Copied {text_path_yt} to {output_text_path}")
            else:
                print(f"No transcript found for keyframe {segment_idx} in either whisper or YT directories.")
            if os.path.isfile(image_path):
                output_image_path = os.path.join(output_dir, keyframe_filename)
                shutil.copy(image_path, output_image_path)
                print(f"Copied {image_path} to {output_image_path}")
        else:
            print(f"No matching segment found in filename: {keyframe_filename}")
            
def format_labels(labels, key):
    return [label.strip() for label in labels[key].replace('\\\n', '').split(',')]
    
def get_model_device(model):
    return next(model.parameters()).device

def model_clip(config_path=config_path):
    model_name = model_config['model_clip']
    model_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_clip = torch.nn.DataParallel(model_clip)
        model_clip = model_clip.to('cuda')
    return model_clip, preprocess_train, preprocess_val, tokenizer

def model_clap(config_path=config_path):
    if not os.path.isfile(model_config['model_clap_checkpoint'].split('/')[-1]):
        subprocess.run(['wget', model_config['model_clap_checkpoint']])
    model_clap = laion_clap.CLAP_Module(enable_fusion=False, amodel=model_config['model_clap'])
    checkpoint = torch.load(model_config['model_clap_checkpoint'].split('/')[-1], map_location='cpu')
    model_clap.load_state_dict(checkpoint, strict=False)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_clap = torch.nn.DataParallel(model_clap)
        model_clap = model_clap.to('cuda')
    return model_clap

def prepare_audio_labels():
    if not os.path.exists('clap-audioset-probe/clap-probe.pkl') or not os.path.exists('clap-audioset-probe/clap-probe.csv'):
        print("Required files not found. Cloning repository...")
        subprocess.run(["git", "clone", "https://github.com/MichaelALong/clap-audioset-probe"])
    with open('clap-audioset-probe/clap-probe.pkl', 'rb') as f:
        multioutput_model = pickle.load(f)
    dfmetrics = pd.read_csv("clap-audioset-probe/clap-probe.csv")
    dfmetrics = dfmetrics.sort_values("model_order")
    model_order_to_group_name = pd.Series(dfmetrics.group_name.values, index=dfmetrics.model_order).to_dict()
    return multioutput_model, model_order_to_group_name, dfmetrics
    
def get_audio_embeddings(audio_path, model_clap):
    print(f"Scanning {audio_path} for audio files...")
    keyframe_pattern = r'keyframe_\d+(_vocals)?\.mp3$'
    audio_files = sorted([
        f for f in glob.glob(audio_path + '/**/*.mp3', recursive=True)
        if re.search(keyframe_pattern, f)])
    print(f"Found {len(audio_files)} audio files matching the patterns.")
    embeddings = []
    for input_file in audio_files:
        print(f"Processing {input_file}...")
        audio_embed = model_clap.get_audio_embedding_from_filelist([input_file], use_tensor=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        audio_embed = audio_embed.to(device)
        normalized_embed = normalize_scores(audio_embed.detach().reshape(1, -1).cpu().numpy())
        embeddings.append(normalized_embed)
    return audio_files, np.vstack(embeddings)

def get_embeddings(model_clip, tokenizer, config_path=config_path):
    
    emotions = format_labels(labels, 'emotions')
    check_if_person_list = format_labels(labels, 'checkifperson')
    number_of_faces_list = format_labels(labels, 'numberoffaces')
    engagement_labels_list = format_labels(labels, 'engagementlabels')
    orientation_labels_list = format_labels(labels, 'orientationlabels')
    check_type_person_list = format_labels(labels, 'checktypeperson')
    valence_list = format_labels(labels, 'valence')
    text_features = generate_embeddings(tokenizer, model_clip, emotions, f"{evaluations['embeddings']}/text_features.npy")
    text_features_if_person = generate_embeddings(tokenizer, model_clip, check_if_person_list, f"{evaluations['embeddings']}/text_features_if_person.npy")
    text_features_type_person = generate_embeddings(tokenizer, model_clip, check_type_person_list, f"{evaluations['embeddings']}/text_features_type_person.npy")
    text_features_if_number_of_faces = generate_embeddings(tokenizer, model_clip, number_of_faces_list, f"{evaluations['embeddings']}/text_features_number_of_faces.npy")
    text_features_orientation = generate_embeddings(tokenizer, model_clip, orientation_labels_list, f"{evaluations['embeddings']}/text_features_orientation.npy")
    text_features_if_engaged = generate_embeddings(tokenizer, model_clip, engagement_labels_list, f"{evaluations['embeddings']}/text_features_if_engaged.npy")
    text_features_valence = generate_embeddings(tokenizer, model_clip, valence_list, f"{evaluations['embeddings']}/text_valence.npy")

def process_files(sample):
    result = {}
    for key, value in sample.items():
        if key.endswith("json"):
            result[key] = json.loads(value)
        elif key.endswith("npy"):
            result[key] = np.load(io.BytesIO(value))
        elif key.endswith(".png"):
            result[key] = Image.open(io.BytesIO(value)).convert("RGB")
        elif key.endswith("m4a") or key.endswith("flac") or key.endswith("mp3"):
            audio_format = "m4a" if key.endswith("m4a") else ("flac" if key.endswith("flac") else "mp3")
            audio_file = io.BytesIO(value)
            try:
                result[key] = AudioSegment.from_file(audio_file, format=audio_format)
            except Exception as e:
                print(f"Error processing {key}: {e}")
        elif key.endswith("txt"):
            try:
                text_content = value.decode('utf-8')
                result[key] = text_content
            except Exception as e:
                print(f"Error decoding {key}: {e}")
    return result

def move_paired(audio_segment, text_content, whisper_segment_dir, segment_key):
    os.makedirs(whisper_segment_dir, exist_ok=True)
    audio_destination_path = os.path.join(whisper_segment_dir, f"{segment_key}.mp3")
    audio_segment.export(audio_destination_path, format="mp3")
    print(f"Copied whisper audio segment to {audio_destination_path}")
    if text_content:
        text_destination_path = os.path.join(whisper_segment_dir, f"{segment_key}.txt")
        with open(text_destination_path, "w", encoding="utf-8") as text_file:
            text_file.write(text_content)
        print(f"Saved associated text file to {text_destination_path}")