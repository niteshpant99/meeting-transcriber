import whisper
import librosa
import os
import argparse
import logging
import sys
import numpy as np

def setup_logging():
    """Sets up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to standard output
        ]
    )

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("input_file", help="Path to the input audio file.")
    parser.add_argument("-o", "--output_dir", default="transcribed_text",
                        help="Directory to save the transcription text file (default: transcribed_text).")
    parser.add_argument("-m", "--model_name", default="large",
                        help="Name of the Whisper model to use (e.g., tiny, base, small, medium, large). Default: large.")
    parser.add_argument("-sr", "--sample_rate", type=int, default=16000,
                        help="Sample rate to resample the audio to (default: 16000).")
    return parser.parse_args()

def load_audio(file_path, sample_rate):
    """
    Loads an audio file using librosa and resamples it.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Target sample rate.

    Returns:
        tuple: A tuple containing the audio data (numpy array) and the original sample rate,
               or (None, None) if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"Input audio file not found: {file_path}")
        return None, None

    try:
        logging.info(f"Loading audio file: {file_path}...")
        # Load audio and automatically resample to the target sample rate
        audio_data, _ = librosa.load(file_path, sr=sample_rate)
        logging.info(f"Audio file loaded and resampled to {sample_rate} Hz successfully.")
        return audio_data, sample_rate
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        return None, None

def transcribe_audio(audio_data, model_name="large"):
    """
    Transcribes audio data using the specified Whisper model.

    Args:
        audio_data (np.ndarray): Numpy array of the audio data.
        model_name (str): Name of the Whisper model to load.

    Returns:
        dict: The transcription result dictionary from Whisper, or None if transcription fails.
    """
    try:
        logging.info(f"Loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name)
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading Whisper model '{model_name}': {e}")
        return None

    try:
        logging.info("Starting transcription...")
        result = model.transcribe(audio_data)
        logging.info("Transcription finished.")
        return result
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

def save_transcription(transcription_text, input_filename, output_dir):
    """
    Saves the transcription text to a file.

    Args:
        transcription_text (str): The text to save.
        input_filename (str): The original name of the input audio file.
        output_dir (str): The directory to save the output file in.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct output filename (e.g., input.m4a -> input.txt)
        base_name = os.path.basename(input_filename)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"{file_name_without_ext}.txt"
        output_path = os.path.join(output_dir, output_filename)

        logging.info(f"Saving transcription to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)
        logging.info("Transcription saved successfully.")
    except Exception as e:
        logging.error(f"Error saving transcription to {output_path}: {e}")

def main(args):
    """Main execution logic."""
    # Ensure input path is treated correctly (relative to CWD or absolute)
    input_file_path = os.path.abspath(args.input_file)

    audio_data, sr = load_audio(input_file_path, args.sample_rate)

    if audio_data is not None:
        transcription_result = transcribe_audio(audio_data, args.model_name)

        if transcription_result and "text" in transcription_result:
            save_transcription(transcription_result["text"], args.input_file, args.output_dir)
        elif transcription_result is None:
            logging.error("Transcription failed.")
        else:
            logging.warning("Transcription result did not contain 'text' key.")
    else:
        logging.error("Audio loading failed. Cannot proceed with transcription.")

if __name__ == "__main__":
    setup_logging()
    parsed_args = parse_arguments()
    main(parsed_args)
