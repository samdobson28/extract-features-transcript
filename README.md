# Audio & Transcript Extraction Pipeline

## Overview
This pipeline extracts **transcripts** and **audio features** from YouTube videos. It processes the audio in **1-second segments**, making it suitable for **machine learning models** that analyze speech patterns, emotions, and linguistic features.

## Features
- **Downloads YouTube audio** in `.wav` format
- **Extracts transcripts** using OpenAI's `whisper` ASR model
- **Segments audio into 1-second intervals**
- **Computes audio features** using `librosa`
- **Saves output** as:
  - Transcript (`.txt` file)
  - Audio features (`.csv` file)

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Install Dependencies
Run the following command to install all required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To process a YouTube video, run:
```bash
python extractor.py
```
This will:
1. Download the **audio** from the provided YouTube URL.
2. Generate a **transcript** using Whisper.
3. Extract **1-second audio feature segments**.
4. Save outputs in the `outputs/` directory.

### Example Output Files:
- `outputs/<video_id>_transcript.txt`  → **Transcript file**
- `outputs/<video_id>_audio_features.csv` → **Audio features per segment**

## File Structure
```
project/
│── extractor.py  # Main script
│── requirements.txt  # Dependencies
│── downloads/  # Downloaded audio files
│── outputs/  # Extracted transcripts & features
```

## Next Steps
1. **Align Features with Empathy Annotations**
   - our dataset includes **empathetic, neutral, and anti-empathetic** labels. These should be **matched to corresponding timestamps** in the segmented audio.
   - Idea: Use **cross-referencing** between extracted segments and existing **annotations** in our dataset.

2. **Feature Engineering**
   - **IMPORTANT**: should use different segment lengths for different features based on past research.
   - Consider additional **speech-related features** such as:
     - **Prosody:** Pitch variation, loudness
     - **Pauses & Silence Detection:** Frequency of pauses
     - **Voice Timbre Features:** Harmonic-to-noise ratio

3. **Model Training Pipeline**
   - Once aligned, these features can be **fed into an ML model** (e.g., LSTM, Transformers) to predict empathy categories.
   - Test **zero-shot** or **few-shot** classification using LLMs.

4. **Data Visualization**
   - Plot feature distributions to **see trends** in speech patterns across empathy labels.
   - Example: Compare **pitch variation** in empathetic vs. neutral segments.

5. **Fine-tuning Whisper for Speech-to-Text**
   - If transcription accuracy needs improvement, consider **fine-tuning Whisper** on our dataset.

## Questions / Issues?
Feel free to tweak the segment length, try different feature extraction methods, or explore alternative ASR models.

---
🚀 **Next step:** Conduct feature analysis, modify feature extraction pipeline based on analysis, begin one and few shots w LLMs. **IMPORTANT**: should use different segment lengths for different features based on past research.
