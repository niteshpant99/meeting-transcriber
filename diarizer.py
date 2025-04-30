import whisperx
import torch
import argparse
import os
import gc
import logging
import sys
import time

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
    parser = argparse.ArgumentParser(description="Transcribe and diarize an audio file using WhisperX.")
    parser.add_argument("input_file", help="Path to the input audio file.")
    parser.add_argument("-o", "--output_dir", default="diarized_text",
                        help="Directory to save the diarized transcription text file (default: diarized_text).")
    parser.add_argument("-m", "--model_name", default="large-v2",
                        help="Name of the Whisper model to use (e.g., tiny, base, small, medium, large-v2, large-v3). Default: large-v2.")
    parser.add_argument("-d", "--device", default=None,
                        help="Device to use for computation ('cpu', 'cuda'). Defaults to 'cuda' if available, else 'cpu'.")
    parser.add_argument("-ct", "--compute_type", default="float16",
                        help="Compute type for Whisper model (e.g., float16, int8). Default: float16.")
    parser.add_argument("-bs", "--batch_size", type=int, default=16,
                        help="Batch size for transcription. Default: 16.")
    parser.add_argument("--hf_token", default=None,
                        help="Hugging Face token for using pyannote.audio models. Reads from HF_TOKEN env var if not provided.")
    parser.add_argument("--language", default=None,
                        help="Language code of the audio (e.g., 'en', 'es'). If None, WhisperX will detect it.")
    return parser.parse_args()

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS.ms"""
    if seconds is None:
        return "??:??:??.???"
    seconds = float(seconds) # Ensure it's a float
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{ms:03}"

def save_diarized_transcription(result, input_filename, output_dir):
    """
    Formats and saves the diarized transcription to a file.

    Args:
        result (dict): The result dictionary containing segments with speaker assignments.
        input_filename (str): The original name of the input audio file.
        output_dir (str): The directory to save the output file in.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(input_filename)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"{file_name_without_ext}_diarized.txt"
        output_path = os.path.join(output_dir, output_filename)

        logging.info(f"Saving diarized transcription to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            # Check if segments exist and are iterable
            if "segments" not in result or not isinstance(result["segments"], list):
                 logging.error("Result dictionary does not contain a valid 'segments' list.")
                 f.write("Error: No valid segments found in the result.\n")
                 return

            for segment in result["segments"]:
                # Check essential keys; provide defaults or skip if missing
                speaker_label = segment.get("speaker", "UNKNOWN_SPEAKER")
                start_time_val = segment.get("start")
                end_time_val = segment.get("end")
                text = segment.get("text", "").strip()

                # Format timestamps safely
                start_time = format_timestamp(start_time_val)
                end_time = format_timestamp(end_time_val)

                if not text: # Skip segments with no text
                    continue

                # Simpler format: one line per segment
                f.write(f"[{start_time} -> {end_time}] {speaker_label}: {text}\n")

        logging.info("Diarized transcription saved successfully.")

    except Exception as e:
        logging.error(f"Error saving diarized transcription to {output_path}: {e}", exc_info=True)


def main(args):
    """Main execution logic for diarization."""
    start_time = time.time()
    logging.info("Starting diarization process...")

    # --- Device Selection ---
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    if device == "cuda" and args.compute_type == "int8" and not torch.cuda.is_bf16_supported():
         logging.warning("int8 compute type is not supported on this GPU. Falling back to float16.")
         args.compute_type = "float16"


    # --- Hugging Face Token ---
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logging.warning("Hugging Face token not provided via argument or HF_TOKEN env var. Diarization might fail if model requires authentication.")
        # Consider raising an error if the token is strictly required for the chosen diarization model
        # raise ValueError("Hugging Face token is required for pyannote.audio models.")

    # --- Load Audio ---
    input_file_path = os.path.abspath(args.input_file)
    if not os.path.exists(input_file_path):
        logging.error(f"Input audio file not found: {input_file_path}")
        return

    audio = None # Initialize audio variable
    try:
        logging.info(f"Loading audio from: {input_file_path}")
        audio = whisperx.load_audio(input_file_path)
        logging.info("Audio loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading audio: {e}", exc_info=True)
        return # Exit if audio loading fails

    # Initialize result variable
    result = None
    model = None
    model_a = None
    metadata = None
    diarize_model = None
    diarize_segments = None

    try:
        # --- Transcription ---
        logging.info(f"Loading Whisper model: {args.model_name} (compute_type: {args.compute_type})")
        model = whisperx.load_model(args.model_name, device, compute_type=args.compute_type)
        logging.info("Transcribing audio...")
        # Pass the language argument if provided
        result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)
        logging.info("Transcription complete.")

        # --- Align Transcription ---
        logging.info("Loading alignment model...")
        # Ensure language code exists in the result before using it
        lang_code = result.get("language", "en") # Default to 'en' if not detected
        logging.info(f"Using language code for alignment: {lang_code}")
        model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
        logging.info("Aligning transcription...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        logging.info("Alignment complete.")

        # --- Diarization ---
        logging.info("Loading diarization pipeline...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        logging.info("Performing speaker diarization...")
        diarize_segments = diarize_model(audio) # Pass the loaded audio
        logging.info("Diarization complete.")

        # --- Assign Speaker Labels ---
        logging.info("Assigning speaker labels to words/segments...")
        # Check if diarize_segments is valid before assignment
        if diarize_segments is not None and not diarize_segments.empty:
             result = whisperx.assign_word_speakers(diarize_segments, result)
             logging.info("Speaker assignment complete.")
        else:
             logging.warning("Diarization produced no segments. Skipping speaker assignment.")
             # Ensure segments still have a placeholder speaker if needed by save function
             for seg in result.get("segments", []):
                 seg["speaker"] = seg.get("speaker", "UNKNOWN")


    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        # Decide whether to save partial results or just report error
        if result and "segments" in result:
            logging.warning("Saving potentially incomplete results due to error.")
            save_diarized_transcription(result, args.input_file, args.output_dir)
        return # Exit after error

    finally:
        # --- Cleanup ---
        logging.info("Cleaning up models and freeing memory...")
        del model
        del model_a
        del metadata
        del diarize_model
        del diarize_segments
        # del audio # Keep audio if needed elsewhere?
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        logging.info("Cleanup complete.")

    # --- Save Final Results ---
    if result:
        save_diarized_transcription(result, args.input_file, args.output_dir)
    else:
        logging.error("No result generated to save.")


    end_time = time.time()
    logging.info(f"Diarization process finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    setup_logging()
    parsed_args = parse_arguments()
    main(parsed_args)
