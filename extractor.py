import os
import yt_dlp
import whisper
import librosa
import numpy as np
import pandas as pd
from pytube import YouTube

def download_audio(youtube_url, output_path="downloads"):
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = f"{output_path}/{info['id']}.wav"
        return filename, info['id']

def extract_transcript(audio_path, video_id, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    transcript_text = result["text"]
    transcript_path = f"outputs/{video_id}_transcript.txt"
    os.makedirs("outputs", exist_ok=True)
    
    with open(transcript_path, "w") as f:
        f.write(transcript_text)
    
    return transcript_path

def extract_audio_features(audio_path, output_csv, segment_length=1.0):
    """
    Extracts audio features from an audio file in fixed-length segments.
    
    :param audio_path: Path to the audio file.
    :param output_csv: Path to save extracted features.
    :param segment_length: Length of each segment in seconds.
    """
    y, sr = librosa.load(audio_path, sr=None)
    segment_samples = int(segment_length * sr)  # Convert segment length to samples

    # Calculate number of segments
    num_segments = len(y) // segment_samples

    features_list = []

    for i in range(num_segments):
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples
        segment = y[start_sample:end_sample]

        # Extract features for this segment
        features = {
            "segment_start_sec": start_sample / sr,
            "segment_end_sec": end_sample / sr,
            "mfcc": librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
            "spectral_centroid": librosa.feature.spectral_centroid(y=segment, sr=sr).mean().tolist(),
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean().tolist(),
            "spectral_contrast": librosa.feature.spectral_contrast(y=segment, sr=sr).mean().tolist(),
            "spectral_rolloff": librosa.feature.spectral_rolloff(y=segment, sr=sr).mean().tolist(),
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(segment).mean().tolist(),
            "rmse": librosa.feature.rms(y=segment).mean().tolist()
        }
        features_list.append(features)

    # Convert to DataFrame and save
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    
    return output_csv

def process_youtube_video(youtube_url):
    print("Downloading audio...")
    audio_path, video_id = download_audio(youtube_url)  # Download only once
    
    print("Extracting transcript...")
    transcript_path = extract_transcript(audio_path, video_id)  # Pass audio file
    
    print("Extracting audio features...")
    output_csv = f"outputs/{video_id}_audio_features.csv"
    feature_path = extract_audio_features(audio_path, output_csv, segment_length=1.0)
    
    print(f"Transcript saved at: {transcript_path}")
    print(f"Audio features saved at: {feature_path}")

# Example usage
youtube_url = "https://www.youtube.com/watch?v=ZNnk9L2LSZI"
process_youtube_video(youtube_url)
