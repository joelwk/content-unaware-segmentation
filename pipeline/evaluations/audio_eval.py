import subprocess
import os
import shutil
import glob
from pydub import AudioSegment
from pydub.utils import mediainfo
import glob
import os
import pickle
import re
import numpy as np
import json

from evaluations.prepare import (
    read_config, get_all_video_ids,generate_embeddings, format_labels, tensor_to_array, 
    remove_duplicate_extension, model_clap, normalize_scores,load_key_image_files,
    load_key_audio_files,get_all_video_ids,prepare_audio_labels,get_audio_embeddings,get_video_ids
)

def get_audio_duration(audio_file):
    audio = AudioSegment.from_file(audio_file)
    return len(audio)

def convert_flac_to_mp3(directory):
    # Find all .flac files in the specified directory
    flac_files = glob.glob(os.path.join(directory, '*.flac'))
    for flac_file in flac_files:
        # Construct the .mp3 filename
        mp3_file = flac_file.replace('.flac', '.mp3')
        # Use ffmpeg to convert the file
        result = subprocess.run(['ffmpeg', '-i', flac_file, '-ab', '320k', '-map_metadata', '0', '-id3v2_version', '3', mp3_file])
        if result.returncode == 0:
            print(f'Successfully converted {flac_file} to {mp3_file}')
            # Remove the original .flac file
            os.remove(flac_file)
            print(f'Removed {flac_file}')
        else:
            print(f'Failed to convert {flac_file}')

def trim_audio(audio_path, max_duration_ms, output_dir):
    duration_ms = get_audio_duration(audio_path)
    file_name = os.path.basename(audio_path)
    trimmed_audio_path = os.path.join(output_dir, file_name)
    if duration_ms > max_duration_ms:
        audio = AudioSegment.from_file(audio_path)
        trimmed_audio = audio[:max_duration_ms]
        trimmed_audio.export(trimmed_audio_path, format='mp3')
        return trimmed_audio_path
    else:
        shutil.copy(audio_path, trimmed_audio_path)
        return trimmed_audio_path

def reorganize_and_move_vocals(processed_dir, model_name='htdemucs'):
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('vocals.mp3') and model_name in root:
                keyframe_id = os.path.basename(root).replace('.mp3', '')
                new_file_name = f"{keyframe_id}_vocals.mp3"
                new_file_path = os.path.join(processed_dir, new_file_name)
                old_file_path = os.path.join(root, file)
                shutil.move(old_file_path, new_file_path)
    for root, dirs, files in os.walk(processed_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")

def separate_audio(input_dir, processed_dir, max_duration_ms, model="htdemucs", extensions=["mp3", "wav", "ogg", "flac"], mp3=True, mp3_rate=320, float32=False, int24=False):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    cmd = ["python3", "-m", "demucs.separate", "-o", str(processed_dir), "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    files = glob.glob(os.path.join(input_dir, '**/*.*'), recursive=True)
    trimmed_files = []
    for file in files:
        if any(file.endswith(ext) for ext in extensions):
            trimmed_path = trim_audio(file, max_duration_ms, processed_dir)
            trimmed_files.append(trimmed_path)
    if not trimmed_files:
        print(f"No valid audio files in {input_dir}")
        return
    p = subprocess.Popen(cmd + trimmed_files, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print("Command failed, something went wrong.")
        print(stderr.decode())
    else:
        print(stdout.decode())

def get_score(file_path):
    """Extracts the 'Human speech' score from the given JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data.get("audio_classification", {}).get("Human speech", 0)
    except FileNotFoundError:
        return 0

def move_specific_file(source_dir, destination_dir, file_name):
    """Moves a specific file from the source to the destination directory."""
    source_path = os.path.join(source_dir, file_name)
    destination_path = os.path.join(destination_dir, file_name)
    if os.path.exists(source_path):
        os.rename(source_path, destination_path)
        print(f"Moved {file_name} to {destination_path}")
    else:
        print(f"File does not exist: {source_path}")

def find_and_move_highest_scoring_files(json_dir, processed_dir):
    processed_clips = set()
    keyframe_pattern = r'keyframe_(\d+?)(_vocals)?\.json$'
    for json_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
        base_name = os.path.basename(json_file)
        keyframe_match = re.search(keyframe_pattern, base_name)
        if keyframe_match and keyframe_match.group(1) not in processed_clips:
            clip_id = keyframe_match.group(1)
            is_vocals = keyframe_match.group(2) is not None
            base_file_name = f'keyframe_{clip_id}'
            regular_file = f'{base_file_name}.json'
            vocals_file = f'{base_file_name}_vocals.json' if is_vocals else regular_file
            regular_score = get_score(os.path.join(json_dir, regular_file))
            vocals_score = get_score(os.path.join(json_dir, vocals_file)) if is_vocals else 0
            chosen_suffix = '_vocals' if vocals_score > regular_score else ''
            files_to_move = [f'{base_file_name}{chosen_suffix}.mp3', f'{base_file_name}{chosen_suffix}.json', f'{base_file_name}{chosen_suffix}_audio_features.npy']
            for file_name in files_to_move:
                move_specific_file(json_dir, os.path.dirname(json_dir), file_name)
            processed_clips.add(clip_id)
      
def zeroshot_classifier_audio(output_dir):
    try:
        params = read_config(section="evaluations")
        model = model_clap()
        multioutput_model, model_order_to_group_name, dfmetrics = prepare_audio_labels()
        audio_files, np_embeddings = get_audio_embeddings(output_dir, model)
        if np_embeddings is None or not np_embeddings.any():
            print("No audio embeddings found.")
            return
        all_probs = multioutput_model.predict_proba(np_embeddings)
        all_probs = np.squeeze(np.dstack(all_probs)[:, 1, :])
        all_preds = np.empty_like(all_probs, dtype=np.int8)
        for _, row in dfmetrics.iterrows():
            order = row["model_order"]
            threshold = row["threshold"]
            all_preds[:, order] = (all_probs[:, order] >= float(params['audio_threshold'])).astype(np.int8)
        for i, input_file in enumerate(audio_files):
            detections = np.where(all_preds[i, :])[0]
            groups_detected = [model_order_to_group_name[x] for x in detections]
            file_name = os.path.basename(input_file)
            file_name = remove_duplicate_extension(file_name)
            filename_without_ext = file_name.split('.')[0]
            save_path = os.path.join(output_dir, filename_without_ext)
            audio_classification = {group: float(all_probs[i, idx]) for idx, group in enumerate(model_order_to_group_name.values())}
            sorted_audio_classification = dict(sorted(audio_classification.items(), key=lambda item: item[1], reverse=True))
            json_data = {
                "audio_path": file_name,
                "audio_classification": sorted_audio_classification}
            json_filename = filename_without_ext + '.json'
            with open(os.path.join(output_dir, json_filename), 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            npy_filename_base = filename_without_ext
            np.save(os.path.join(output_dir, npy_filename_base + '_audio_features.npy'), np_embeddings[i])
    except Exception as e:
        print(f"An error occurred in zeroshot_classifier_audio for video {output_dir.split('/')[-1]}: {e}")

def main():
    params = read_config(section="evaluations")
    config_params = read_config(section="config_params")
    if config_params['mode'] == 'directory':
        video_ids = get_all_video_ids(params['completedatasets'])
    else:
        config_params['mode'] == 'wds'
        video_ids = get_video_ids('./evaluations/image_evaluations/')
    for video in video_ids:
        try:
            in_path = f"./evaluations/image_audio_pairs/{str(video)}"
            output_path = f"./evaluations/audio_evaluations/{str(video)}/audio_processed"
            max_duration_ms = int(params['max_duration_ms'] * 1000)
            final_audio = f"./evaluations/audio_evaluations/{str(video)}/"
            convert_flac_to_mp3(output_path)
            separate_audio(in_path, output_path, max_duration_ms)
            reorganize_and_move_vocals(output_path)
            convert_flac_to_mp3(output_path)
            zeroshot_classifier_audio(output_path)
            find_and_move_highest_scoring_files(output_path, final_audio)
        except IndexError as e:
            print(f"Index error occurred for video {video}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for video {video}: {e}")
            #TODO: improve logic for handling errors
           # shutil.rmtree(f"{params['completedatasets']}/{str(video)}")
            shutil.rmtree(output_path)
            continue
    print("All videos processed.")
if __name__ == '__main__':
    main()