from PIL import Image
import os
import shutil
import json
import subprocess
import argparse
import numpy as np
import glob
import torch
import webdataset as wds
import json
import numpy as np
from PIL import Image
import io
import glob
import os
import re
from webdataset import ShardWriter
from pydub import AudioSegment
from io import BytesIO
import ast
import warnings

from evaluations.prepare import (
    read_config, generate_embeddings, format_labels,remove_duplicate_extension, process_keyframe_audio_pairs, 
    get_embeddings, model_clip, normalize_scores, softmax, sort_and_store_scores, load_key_image_files, 
    load_key_audio_files, get_all_video_ids,process_files, move_paired
)
    
def is_good_image(is_person, face_probs, orientation_probs, engagement_probs):
    thresholds = read_config(section="thresholds")
    # Define thresholds
    is_person_threshold = float(thresholds['is_person_threshold']) # High probability of the subject being a person
    single_face_threshold = float(thresholds['single_face_threshold'])  # High probability of there being only one face
    facing_forward_threshold = float(thresholds['facing_forward_threshold'])  # High probability of the subject facing forward
    engagement_threshold = float(thresholds['engagement_threshold'])  # High probability of the subject looking at the camera or not, depending on preference
    type_person_threshold = float(thresholds['type_person_threshold'])  # Threshold for type of person detection

    # Check conditions
    is_person_detected = is_person[1] > is_person_threshold
    single_face_detected = face_probs[0] > single_face_threshold
    facing_forward = orientation_probs[0] > facing_forward_threshold
    engaged = engagement_probs[0] > engagement_threshold or engagement_probs[1] > engagement_threshold
    # Return True if the image meets the criteria for being "Good"
    return is_person_detected and single_face_detected and facing_forward and engaged

def zeroshot_classifier(image_path, video_identifier, output_dir, key=None):
    if key is None:
        key = image_path
        image_path = Image.open(image_path)
        output_dir = os.path.join(output_dir, str(video_identifier))

    params = read_config(section="evaluations")
    labels = read_config("labels")
    model, preprocess_train, preprocess_val, tokenizer = model_clip()
    get_embeddings(model, tokenizer)

    # Form the paths to the embeddings
    text_features_path = os.path.join(params['embeddings'], 'text_features.npy')
    text_features_if_person_path = os.path.join(params['embeddings'], 'text_features_if_person.npy')
    text_features_type_person_path = os.path.join(params['embeddings'], 'text_features_type_person.npy')
    text_features_if_number_of_faces_path = os.path.join(params['embeddings'], 'text_features_number_of_faces.npy')
    text_features_orientation_path = os.path.join(params['embeddings'], 'text_features_orientation.npy')
    text_features_if_engaged_path = os.path.join(params['embeddings'], 'text_features_if_engaged.npy')
    text_features_valence_path = os.path.join(params['embeddings'], 'text_valence.npy')

    # Load embeddings
    text_features = np.load(text_features_path)
    text_features_if_person = np.load(text_features_if_person_path)
    text_features_type_person = np.load(text_features_type_person_path)
    text_features_if_number_of_faces = np.load(text_features_if_number_of_faces_path)
    text_features_orientation = np.load(text_features_orientation_path)
    text_features_if_engaged = np.load(text_features_if_engaged_path)
    text_features_valence = np.load(text_features_valence_path)
    
    # Set up the output directory for processed images
    run_output_dir = os.path.join(output_dir)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Load and preprocess the image
    image_preprocessed = preprocess_val(image_path).unsqueeze(0)

    # Encode the image using the CLIP model and normalize the features
    image_preprocessed = image_preprocessed.to('cuda' if torch.cuda.is_available() else 'cpu')
    image_features = model.encode_image(image_preprocessed)
    image_features = image_features.detach().cpu().numpy()
    image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
    
    # Calculate probabilities for different categories using softma
    is_person_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_if_person.T))
    type_person_probs = softmax(float(params['scalingfactor']) * image_features @ text_features_type_person.T)
    face_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_if_number_of_faces.T))
    orientation_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_orientation.T))
    engagement_probs = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features_if_engaged.T))
    text_probs_emotions = softmax(float(params['scalingfactor']) * normalize_scores(image_features @ text_features.T))
    text_score_emotions = image_features @ text_features.T
    text_probs_valence = softmax(float(params['scalingfactor']) * image_features @ text_features_valence.T)
    perform_face_check = params.get('face_detected_in_video_or', 'False').lower() == 'true'
    face_detected = True
    if not perform_face_check:
        face_detected = is_good_image(is_person_probs[0], face_probs[0], orientation_probs[0], engagement_probs[0])
    filename = os.path.basename(key)
    filename_without_ext = filename.split('.')[0]
    filename = remove_duplicate_extension(filename)
    if len(filename.split('_')) > 2:
        filename = filename.split('_')[0] + '_' + filename.split('_')[1] + '.png'
    save_path = os.path.join(run_output_dir, filename)
    image_path.save(save_path)
    sorted_type_person_scores = sort_and_store_scores(type_person_probs[0], format_labels(labels, 'checktypeperson'))
    sorted_emotions = sort_and_store_scores(text_probs_emotions[0], format_labels(labels, 'emotions'))
    sorted_emotions_scores = sort_and_store_scores(text_score_emotions[0], format_labels(labels, 'emotions'))
    sorted_valence = sort_and_store_scores(text_probs_valence[0], format_labels(labels, 'valence'))
    face_detected_python_bool = bool(face_detected)
    json_data = {
            "image_path": filename,
            "face_detected": face_detected_python_bool,
            "face_detection_scores": sorted_type_person_scores,
            "emotions_probs": sorted_emotions,
            "emotions_scores":sorted_emotions_scores,
            "valence": sorted_valence}
    filename_without_ext = filename_without_ext.split('_')[0] + '_' + filename_without_ext.split('_')[1]
    json_filename = filename_without_ext + '.json'
    with open(os.path.join(run_output_dir, json_filename), 'w') as json_file:
      json.dump(json_data, json_file, indent=4)
    npy_filename_base = filename_without_ext
    np.save(os.path.join(run_output_dir, npy_filename_base + '_image_features.npy'), image_features)
    return face_detected_python_bool

def process_from_directory():
    params = read_config(section="evaluations")
    video_ids = get_all_video_ids(params['completedatasets'])
    for video in video_ids:
        try:
            face_detected_in_video = params.get('face_detected_in_video_or', 'False').lower() == 'true'
            keyframes = load_key_image_files(video, params)
            for keyframe in keyframes:
                if zeroshot_classifier(keyframe, video, os.path.join(params['outputs'], "image_evaluations"), key=None):
                    face_detected_in_video = True
                if not face_detected_in_video:  
                    video_dir = os.path.join(params['outputs'], "image_evaluations", str(video))
                    if os.path.exists(video_dir):
                        shutil.rmtree(video_dir)
                        video__original_dir = os.path.join(params['completedatasets'], str(video))
                        shutil.rmtree(video__original_dir)
                        print(f"No faces detected in any keyframes of video {video}. Directory {video_dir} removed.")
                    continue
            image_dir = os.path.join(params['outputs'], "image_evaluations", str(video))
            output_dir = os.path.join(params['outputs'], "image_audio_pairs", str(video))
            audio_dir = os.path.join(params['completedatasets'], str(video), "keyframe_audio_clips")
            process_keyframe_audio_pairs(image_dir, audio_dir, output_dir)
        except Exception as e:
            print(f"Failed to process images and pair with audio for video {video}: {e}")

def process_from_wds():
    params = read_config(section="evaluations")
    dataset_paths =  glob.glob(f"{params['wds_dir']}/completed_datasets-*.tar")
    dataset = wds.WebDataset(dataset_paths).map(process_files)
    whisper_segments = {}
    text_segments = {}
    #TODO: Add checks to avoid reproreprocessing
    for sample in dataset:
        video_id = sample['__key__'].split('/')[0]
        image_dir = os.path.join(params['outputs'], "image_evaluations", video_id)
        output_dir = os.path.join(params['outputs'], "image_audio_pairs", video_id)
        face_detected_in_video = params.get('face_detected_in_video_or', 'False').lower() == 'true'
        for key, value in sample.items():
            if key.endswith('mp3') and isinstance(value, AudioSegment):
                segment_key = sample['__key__']
                segment_match = re.search(r'keyframe_(\d+)', segment_key)
                if segment_match:
                    segment_id = str(segment_match.group(1))
                    whisper_segments[segment_id] = value
            if key.endswith('txt'):
                segment_key = sample['__key__']
                segment_match = re.search(r'keyframe_(\d+)', segment_key)
                if segment_match:
                    segment_id = str(segment_match.group(1))
                    try:
                        text_content = value
                        text_segments[segment_id] = text_content
                    except UnicodeDecodeError as e:
                        print(f"Error decoding text for segment {segment_id}: {e}")
            if key.endswith('png') and isinstance(value, Image.Image):
                keyframe_match = re.search(r'keyframe_(\d+)_timestamp', sample['__key__'])
                if keyframe_match:
                    keyframe_id = str(keyframe_match.group(1))
                    keyframe_filename = f"keyframe_{keyframe_id}.png"
                    if zeroshot_classifier(value, video_id, image_dir, key=keyframe_filename):
                        face_detected_in_video = True
                        if keyframe_id in whisper_segments:
                            audio_segment = whisper_segments[keyframe_id]
                            text_content = text_segments.get(keyframe_id, "")
                            move_paired(audio_segment, text_content, output_dir, f"keyframe_{keyframe_id}")
                            paired_png = os.path.join(image_dir, keyframe_filename)
                            shutil.copy(paired_png, output_dir)
                            print(f"Paired keyframe {str(keyframe_id)} with its whisper segment and text")
                            
def main():
    params = read_config(section="config_params")
    if params['mode'] == 'directory':
        process_from_directory()
    elif params['mode'] == 'wds':
        process_from_wds()

if __name__ == '__main__':
    main()