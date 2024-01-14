import os
import sys
import json
import subprocess
import glob
import warnings
from pipeline import read_config, string_to_bool

warnings.filterwarnings("ignore", category=UserWarning)

def read_keyframe_data(keyframe_json_path):
    with open(keyframe_json_path, 'r') as file:
        return json.load(file)
        
def install_requirements():
    print("Installing required packages and restarting...")
    subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio"])
    subprocess.run(["pip", "install", "accelerate", "optimum"])
    subprocess.run(["pip", "install", "ipython-autotime"])
    subprocess.run(["pip", "install", "pydub"])
    subprocess.run(["pip", "install","transformers"])
    
install_requirements()
import torch
from pydub import AudioSegment
from transformers import pipeline

evaluations = read_config(section="evaluations")    
config_params = read_config(section="config_params")

def convert_audio_files(input_directory, output_directory, output_format="flac"):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.endswith(".m4a"):
            m4a_path = os.path.join(input_directory, filename)
            output_filename = os.path.splitext(filename)[0] + f".{output_format}"
            output_path = os.path.join(output_directory, output_filename)
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Overwriting.")
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio.export(output_path, format=output_format)
            print(f"Converted {m4a_path} to {output_path}")

def segment_audio_using_keyframes(audio_path, audio_clip_output_dir, keyframe_data, duration, suffix_=None):
    os.makedirs(audio_clip_output_dir, exist_ok=True)
    duration = int(duration/1000)
    output_aligned = [{'segment_idx': idx, 'timestamp': [keyframe['time_frame'], keyframe['time_frame'] + duration]} for idx, keyframe in keyframe_data.items()]
    for segment in output_aligned:
        start_time = segment['timestamp'][0]
        adjusted_start_time = start_time
        suffix_str = f"_{suffix_}" if suffix_ else ""
        output_segment_path = f"{audio_clip_output_dir}/keyframe_{segment['segment_idx']}{suffix_str}.flac"
        command = [
            'ffmpeg',
            '-ss', str(adjusted_start_time),
            '-t', str(duration),
            '-i', audio_path,
            '-acodec', 'flac',
            '-y', output_segment_path
        ]
        subprocess.run(command, check=True)
    json_path = os.path.join(audio_clip_output_dir, 'keyframe_timestamps.json')
    with open(json_path, 'w') as f:
        json.dump(output_aligned, f)

def audio_pipeline(audio_path, audio_clip_output_dir, keyframe_data, duration):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_path)
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline(evaluations['pipeline_function'],
                        evaluations['whisper_model'],
                        torch_dtype=torch.float16,
                        device=device)
        output_aligned_final = []
        for idx, keyframe in keyframe_data.items():
            start_time_ms = keyframe['time_frame'] * 1000  # Convert start time to milliseconds
            end_time_ms = start_time_ms + (duration)  # Calculate end time in milliseconds
            audio_segment = audio[start_time_ms:end_time_ms]
            temp_path = os.path.join(audio_clip_output_dir, f'temp_segment_{idx}.flac')
            audio_segment.export(temp_path, format='flac')
            outputs = pipe(temp_path, return_timestamps=True)
            os.remove(temp_path)  # Remove the temporary file
            chunks = outputs.get("chunks", [])
            if chunks:
                transcript = ' '.join(chunk.get('text', '') for chunk in chunks)
                segment_info = {
                    'segment_idx': idx,
                    'timestamp': [start_time_ms / 1000, end_time_ms / 1000],  
                    'text': transcript}
                output_aligned_final.append(segment_info)
        json_path = os.path.join(audio_clip_output_dir, 'outputs.json')
        with open(json_path, 'w') as f:
            json.dump(output_aligned_final, f)
    except Exception as e:
        print(f"Error in audio_pipeline: {e}")

def full_audio_transcription_pipeline(audio_path, output_dir):
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline(evaluations['pipeline_function'],
                        evaluations['whisper_model'],
                        torch_dtype=torch.float16,
                        device=device)
        outputs = pipe(audio_path,chunk_length_s=int(evaluations['chunk_length']),batch_size=int(evaluations['batch_size']), return_timestamps=True)
        chunks = outputs.get("chunks", [])
        if not chunks:
            print(f"No chunks returned by the pipeline for {audio_path}.")
            return
        transcript = ' '.join(chunk.get('text', '') for chunk in chunks)
        full_transcript_path = os.path.join(output_dir, 'full_transcript.json')
        with open(full_transcript_path, 'w') as f:
            json.dump({'transcript': transcript}, f)
        print(f"Full transcript created: {full_transcript_path}")
    except Exception as e:
        print(f"Error in full_audio_transcription_pipeline: {e}")

def time_to_seconds(timestr):
    h, m, s = timestr.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def write_transcripts_for_keyframes(transcripts, keyframe_timestamps, yt_output_dir):
    if not os.path.exists(yt_output_dir):
        os.makedirs(yt_output_dir)
    for keyframe in keyframe_timestamps:
        segment_idx = keyframe["segment_idx"]
        start_time, end_time = keyframe["timestamp"]
        start_time_sec, end_time_sec = start_time, end_time
        associated_transcripts = []
        for transcript in transcripts:
            transcript_start = time_to_seconds(transcript['start'])
            transcript_end = time_to_seconds(transcript['end'])
            if start_time_sec <= transcript_start < end_time_sec:
                associated_transcripts.append(' '.join(transcript['lines']))
        with open(f'{yt_output_dir}/keyframe_{segment_idx}_yt_transcripts.txt', 'w') as f:
            for line in associated_transcripts:
                f.write(f"{line}\n")
        print(f"Transcripts for keyframe {segment_idx} written to {yt_output_dir}/keyframe_{segment_idx}_yt_transcripts.txt")

def process_audio_files():
    base_path = evaluations['completedatasets']
    for video_dir in os.listdir(base_path):
        n = video_dir
        initial_input_directory = os.path.join(base_path, n, 'originalvideos')
        audio_clip_output_dir = os.path.join(base_path, n, 'keyframe_audio_clips')
        full_audio_clip_output_dir = os.path.join(audio_clip_output_dir, 'whisper_audio_segments', 'full_whisper_segments')
        keyframe_dir = os.path.join(base_path, n, 'keyframes')
        keyframe_json_path = os.path.join(keyframe_dir, 'keyframe_data.json')
        if not os.path.exists(keyframe_json_path):
            print(f"Keyframe data not found for video {n}. Skipping...")
            continue
        keyframe_data = read_keyframe_data(keyframe_json_path)
        for audio_file in os.listdir(initial_input_directory):
            if audio_file.endswith('.m4a'):
                audio_path = os.path.join(initial_input_directory, audio_file)
                process_individual_audio_file(audio_file, audio_path, initial_input_directory, audio_clip_output_dir, keyframe_data, evaluations, n, config_params)
        process_full_audio = string_to_bool(config_params.get("full_whisper_audio", "False"))
        if process_full_audio and os.path.exists(audio_path): 
            process_entire_audio(audio_path, full_audio_clip_output_dir, evaluations)

def process_individual_audio_file(audio_file,audio_path,initial_input_directory, audio_clip_output_dir, keyframe_data, evaluations, video_id, config_params):
    whisper_output_dir = os.path.join(audio_clip_output_dir, 'whisper_audio_segments')
    yt_output_dir = os.path.join(audio_clip_output_dir, 'yt_audio_segments')
    segment_audio_using_keyframes(audio_path, whisper_output_dir, keyframe_data, int(evaluations['max_duration_ms']), suffix_=None)
    individual_output_dir = os.path.join(whisper_output_dir, os.path.splitext(audio_file)[0])
    if not os.path.exists(individual_output_dir):
        os.makedirs(individual_output_dir)
    convert_audio_files(initial_input_directory, individual_output_dir)
    flac_file = os.path.splitext(audio_file)[0] + '.flac'
    full_audio_path = os.path.join(individual_output_dir, flac_file)
    mode = config_params['transcript_mode']
    if mode in ["whisper", "all"]:
        audio_pipeline(full_audio_path, individual_output_dir, keyframe_data, int(evaluations['max_duration_ms']))
    if mode in ["yt", "all"]:
        keyframe_timeframes_path = os.path.join(evaluations['completedatasets'], video_id, 'keyframe_audio_clips/whisper_audio_segments/keyframe_timestamps.json')
        yt_transcripts_path = os.path.join(evaluations['completedatasets'], video_id, f'originalvideos/{video_id}.json')
        if os.path.exists(yt_transcripts_path) and os.path.exists(keyframe_timeframes_path):
            with open(yt_transcripts_path, 'r') as f:
                yt_transcripts = json.load(f)
                if 'yt_meta_dict' in yt_transcripts and 'subtitles' in yt_transcripts['yt_meta_dict']:
                    yt_transcripts = yt_transcripts['yt_meta_dict']['subtitles']
                    with open(keyframe_timeframes_path, 'r') as kf:
                        keyframe_timeframes = json.load(kf)
                    write_transcripts_for_keyframes(yt_transcripts, keyframe_timeframes, yt_output_dir)
                else:
                    print(f"Unexpected JSON structure in {yt_transcripts_path}")
        else:
            print(f"Required files for YT transcripts not found. Check paths: {yt_transcripts_path}, {keyframe_timeframes_path}")

def process_entire_audio(audio_path, full_audio_clip_output_dir, evaluations):
    if not os.path.exists(full_audio_clip_output_dir):
        os.makedirs(full_audio_clip_output_dir)
    full_audio_transcription_pipeline(audio_path, full_audio_clip_output_dir)


def main():
    process_audio_files()

if __name__ == '__main__':
    main()