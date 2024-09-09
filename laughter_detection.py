import os
import whisper

import numpy as np
import pandas as pd

# Audio
from pydub import AudioSegment
from scipy.io import wavfile

# Video
from moviepy.editor import VideoFileClip

from scipy.fft import fft


class LaughterDetection:
    def __init__(self, output_dir: str) -> None:
        self.model = whisper.load_model("large-v2")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.chunks_path = f"{self.output_dir}/chunks"
        os.makedirs(self.chunks_path, exist_ok=True)
        self.transcript_path = f"{self.output_dir}/transcript.csv"

    def download_transcription(self, path: str):
        result = self.model.transcribe(path)
        result = result["segments"]

        df = pd.DataFrame(result)
        df.drop(
            columns=[
                "id",
                "seek",
                "tokens",
                "temperature",
                "avg_logprob",
                "compression_ratio",
            ],
            axis=1,
            inplace=True,
        )
        return df

    def get_transcription(self):
        df = pd.read_csv(self.transcript_path)
        df.drop(columns="Unnamed: 0", axis=1, inplace=True)
        return df

    """
    ============================
    Chunks    
    ============================
    """

    def create_chunks(self, transcripts: pd.DataFrame):
        audio = AudioSegment.from_file("./temp_audio.wav", format="wav")
        chunk = 1
        for i, row in transcripts.iterrows():
            start_ms = row["start"] * 1000
            end_ms = row["end"] * 1000
            # SLice the audio
            audio_segment = audio[start_ms:end_ms]
            # Export the segment.
            output_file = f"chunk{chunk}.wav"
            audio_segment.export(f"{self.chunks_path}/{output_file}", format="wav")
            chunk += 1

    def get_chunks_decibel(self):

        chunk_files = os.listdir(self.chunks_path)
        chunk_index = 1
        chunk_data = {"chunk": [], "dB": [], "maxVol": []}
        for i in chunk_files:
            file_path = f"{self.chunks_path}/{i}"
            audio = AudioSegment.from_file(file_path, format="wav")
            # Get the raw audio data as an array
            samples = np.array(audio.get_array_of_samples())

            # Calculate the mean square of the samples
            mean_square = np.mean(samples**2)

            # Calculate the RMS (Root Mean Square) value
            rms = np.sqrt(mean_square)

            # Convert the RMS value to decibels
            decibels = 20 * np.log10(rms / (2**15))

            maxVol = self.get_max_wav_volume(file_path)
            chunk_data["chunk"].append(chunk_index)
            chunk_data["dB"].append(decibels)
            chunk_data["maxVol"].append(maxVol)
            chunk_index += 1

        df = pd.DataFrame(chunk_data)

        print(f"DF: {df}")

    def get_max_wav_volume(self, path: str):
        sample_rate, data = wavfile.read(path)
        # Check if the audio is stereo or mono
        if len(data.shape) == 2:  # Stereo
            # Calculate the maximum value across both channels
            max_amplitude = np.max(np.abs(data), axis=0).max()
        else:  # Mono
            # Calculate the maximum value in the mono channel
            max_amplitude = np.max(np.abs(data))
        return max_amplitude

    def extract_audio(self, video_path: str):
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio.export("temp_audio.wav", format="wav")

    """
    ============================
    Video Processing    
    ============================
    """

    def process_video(self, video_path: str):

        if not self.does_file_exist("temp_audio.wav"):
            self.extract_audio(video_path)
        # Transcript handling
        df = pd.read_csv(self.transcript_path)
        if df.empty:
            df = self.get_transcription(video_path)
            df.to_csv(self.transcript_path)
        else:
            df.drop(columns="Unnamed: 0", axis=1, inplace=True)
        print(f"DF: {df}")

        if not self.is_dir_filled(self.chunks_path):
            self.create_chunks(df)

        self.get_chunks_decibel()

    def does_file_exist(self, path):
        if os.path.exists(path):
            return True
        else:
            return False

    def is_dir_filled(self, path_to_dir: str):
        dirs = os.listdir(path_to_dir)
        if len(dirs) > 0:
            return True
        else:
            return False

    def _time_to_milliseconds(self, t) -> int:
        """
        Function to convert HH:MM:SS.MMM to milliseconds

        t: str
            String of a timestamp.

        returns: int
            Integer representing the timestamp in milliseconds.
        """
        h, m, s = t.split(":")
        s, ms = s.split(".")
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

    def _milliseconds_to_time(self, ms):
        """
        Function to convert HH:MM:SS.MMM to milliseconds

        ms: int
            Integer representing a timestamp in milliseconds.

        returns: str
            String representing 'ms' as a timestamp.
        """
        # Calculate hours, minutes, and seconds
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        # Format the time as HH:MM:SS.sss
        time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"
        return time_str
