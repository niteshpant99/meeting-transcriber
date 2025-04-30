# Audio Transcription & Diarization Tool

This project provides Python scripts for transcribing and diarizing audio files locally.

-   `transcriber.py`: Performs basic audio transcription using the original OpenAI Whisper library.
-   `diarizer.py`: Performs transcription *and* speaker diarization using the WhisperX library (which utilizes Whisper, `pyannote.audio`, and alignment models).

## 1. Basic Transcription (`transcriber.py`)

This script uses the `openai-whisper` library for straightforward transcription.

### Actions

1.  Loads an audio file (various formats supported by `librosa`).
2.  Resamples the audio to 16kHz.
3.  Loads a specified Whisper model (e.g., tiny, base, small, medium, large).
4.  Transcribes the audio content.
5.  Saves the resulting transcription text to a `.txt` file.

### Requirements for `transcriber.py`

-   Python 3.x
-   `ffmpeg` installed on your system (e.g., `brew install ffmpeg` on macOS, `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu).
-   Python packages: `openai-whisper`, `librosa`, `numpy`. You can install these specific packages if you only need basic transcription. See `requirements.txt` for combined dependencies.

### Usage (`transcriber.py`)

```bash
python transcriber.py <input_file> [options]
```

**Arguments:**

*   `<input_file>`: (Required) Path to the input audio file.

**Options:**

*   `-o` or `--output_dir`: Directory to save the transcription text file. Defaults to `transcribed_text`.
*   `-m` or `--model_name`: Name of the Whisper model (`tiny`, `base`, `small`, `medium`, `large`). Defaults to `large`.
*   `-sr` or `--sample_rate`: Sample rate to resample audio to. Defaults to `16000`.

**Example:**

```bash
python transcriber.py audio/call_0.m4a -o transcripts -m base
```

---

## 2. Transcription with Speaker Diarization (`diarizer.py`)

This script uses the `WhisperX` library to perform transcription, word alignment, and speaker diarization.

### Actions

1.  Loads an audio file.
2.  Transcribes the audio using a specified Whisper model (via WhisperX).
3.  Aligns the transcription to get word-level timestamps.
4.  Performs speaker diarization using `pyannote.audio` (via WhisperX) to identify speaker segments.
5.  Assigns speaker labels to the transcribed words/segments.
6.  Saves the formatted, diarized transcription to a `.txt` file.

### Requirements for `diarizer.py`

-   Python 3.x
-   `ffmpeg` installed on your system.
-   **PyTorch:** Install a version compatible with your system (CPU/CUDA) from [pytorch.org](https://pytorch.org/). WhisperX relies heavily on PyTorch.
-   **Hugging Face Account & Token:** Diarization models from `pyannote.audio` often require authentication.
    1.  Create a Hugging Face account if you don't have one.
    2.  Accept the user conditions for the relevant `pyannote` models on the Hugging Face Hub (e.g., `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`). You might need to do this by visiting their model pages on the Hub.
    3.  Create an access token with 'read' permissions at [hf.co/settings/tokens](https://hf.co/settings/tokens).
    4.  Make the token available to the script, either by:
        *   Logging in via the terminal: `huggingface-cli login` (recommended)
        *   Setting the `HF_TOKEN` environment variable.
        *   Passing the token via the `--hf_token` command-line argument (less secure).
-   **Python Packages:** Install all dependencies using `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs `torch`, `pyannote.audio`, `whisperx` from GitHub, `librosa`, and `numpy`.*

### Usage (`diarizer.py`)

```bash
python diarizer.py <input_file> [options]
```

**Arguments:**

*   `<input_file>`: (Required) Path to the input audio file.

**Options:**

*   `-o` or `--output_dir`: Directory to save the diarized transcription. Defaults to `diarized_text`.
*   `-m` or `--model_name`: Whisper model name (e.g., `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`). Defaults to `large-v2`.
*   `-d` or `--device`: Compute device (`cpu`, `cuda`). Defaults to `cuda` if available.
*   `-ct` or `--compute_type`: Model compute type (`float16`, `int8`, `float32`). Defaults to `float16`. `int8` requires compatible hardware.
*   `-bs` or `--batch_size`: Transcription batch size. Defaults to `16`. Adjust based on VRAM.
*   `--hf_token`: Your Hugging Face access token (if not logged in or using env var).

**Example:**

```bash
# Make sure you are logged in via huggingface-cli login first!
python diarizer.py audio/call_1.m4a -m medium -d cuda -ct float16
```

This command will:
*   Load `audio/call_1.m4a`.
*   Use the `medium` Whisper model on a CUDA device with `float16` precision.
*   Perform transcription, alignment, and diarization.
*   Save the output to `diarized_text/call_1_diarized.txt`.

**Memory Management:**

WhisperX can be memory-intensive, especially with large models or long audio files on GPUs. The `diarizer.py` script includes basic cleanup (`gc.collect()`, `torch.cuda.empty_cache()`). If you encounter out-of-memory errors, try:
*   Using a smaller Whisper model (`-m`).
*   Reducing the batch size (`-bs`).
*   Using a more memory-efficient compute type (`-ct int8` if supported).
*   Running on the CPU (`-d cpu`), which will be significantly slower.
