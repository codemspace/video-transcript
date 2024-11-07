import os
import time
import logging
import whisper_timestamped as whisper
from moviepy.editor import VideoFileClip

from config import (
    INPUT_VIDEOS_DIR,
    OUTPUT_VIDEOS_DIR,
    MODEL_NAME,
    LANGUAGE,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SRTGenerator:
    def __init__(self, video_file: str):
        """Initialize the SRT generator with a video file path."""
        self.video_file = video_file

    def generate_srt(self):
        """Generate SRT file for the provided video."""
        # Load the video and extract audio
        clip = VideoFileClip(self.video_file)
        audio = clip.audio
        temp_audio_path = "temp_audio.mp3"
        audio.write_audiofile(temp_audio_path, codec="mp3", verbose=False, logger=None)

        # Transcribe audio using Whisper
        loaded_audio = whisper.load_audio(temp_audio_path)
        model = whisper.load_model(MODEL_NAME, device="cpu")
        result = whisper.transcribe(model, loaded_audio, language=LANGUAGE, verbose=None)

        # Clean up the temporary audio file
        os.remove(temp_audio_path)

        # Extract transcription segments for SRT
        srt_content = self.format_srt(result['segments'])

        # Save the SRT file
        srt_file_path = os.path.join(OUTPUT_VIDEOS_DIR, os.path.basename(self.video_file).replace(".mp4", ".srt"))
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        logging.info(f"SRT file generated: {srt_file_path}")

    def format_srt(self, segments):
        """Format transcription segments as an SRT string."""
        srt_content = ""
        for i, segment in enumerate(segments):
            start_time = self.format_timestamp(segment['start'])
            end_time = self.format_timestamp(segment['end'])
            text = segment['text']
            srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"
        return srt_content
    
    @staticmethod
    def format_timestamp(seconds):
        """Convert a timestamp in seconds to SRT format (hh:mm:ss,ms)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

    # Process each video in the input directory
    for video_file in os.listdir(INPUT_VIDEOS_DIR):
        if video_file.endswith(".mp4"):  # Adjust if needed for different video formats
            logging.info(f"Processing {video_file}")
            generator = SRTGenerator(os.path.join(INPUT_VIDEOS_DIR, video_file))
            generator.generate_srt()

if __name__ == "__main__":
    main()
