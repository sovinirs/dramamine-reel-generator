# Dramamine Style Reel Generator

A web application that generates stylized Instagram-style reels from audio files with synchronized text overlays. The application supports multiple languages including English, Hindi, and Tamil.

## Features

- Audio file upload support (MP3, WAV, M4A, MP4)
- Multi-language support (English, Hindi, Tamil)
- Automatic speech-to-text transcription
- Romanization of non-English text
- Dynamic text overlay with stylized animations
- Background music integration
- Moody video filters and effects
- Downloadable processed videos

## Prerequisites

- Python 3.x
- OpenAI API key
- Flask
- MoviePy
- Required fonts in `./fonts` directory
- Background videos in `./videos` directory
- Background music (`DramamineFM.mp3`) in `./audio` directory

## Installation

1. Clone the repository:

bash
git clone <repository-url>
cd dramamine-reel-generator

2. Install required dependencies:

```bash
pip install flask openai python-dotenv moviepy
```

3. Create a `.env` file in the root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

4. Add required assets:

- Place your font files (TTF/OTF) in the `./fonts` directory
- Add background videos (MP4) to the `./videos` directory

## Usage

1. Start the Flask server:

```bash
python server.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload an audio file and select the language

4. Wait for processing to complete

5. Download your generated video

## How It Works

1. **Audio Processing**: The application uses OpenAI's Whisper API to transcribe audio files with precise word-level timestamps.

2. **Text Processing**: For non-English content, the text is romanized using OpenAI's language models.

3. **Video Generation**:
   - Text is segmented into natural phrases
   - Random background videos are selected from the video pool
   - Dynamic text overlays are created with random fonts and positions
   - Vignette filters and effects are applied
   - Background music is mixed with the original audio

## Project Structure

- `server.py`: Flask server handling file uploads and processing
- `script.py`: Core processing logic for video generation
- `templates/index.html`: Web interface
- `uploads/`: Temporary storage for uploaded audio files
- `output/`: Generated video storage
- `videos/`: Background video clips
- `fonts/`: Custom fonts for text overlays
- `audio/`: Background music

## Limitations

- Maximum file size: 16MB
- Supported audio formats: MP3, WAV, M4A, MP4
- Supported languages: English, Hindi, Tamil
