# PDF to Podcast Converter

A web service that converts PDF documents into natural-sounding podcast audio files using AI-powered text transformation and speech synthesis.

## Features

- **PDF Text Extraction**: Automatically extracts text content from uploaded PDF files
- **AI-Powered Script Generation**: Transforms technical content into conversational podcast scripts using Claude 3.5 Sonnet
- **Text-to-Speech Synthesis**: Converts scripts to natural-sounding audio using Amazon Titan TTS
- **Asynchronous Processing**: Handles large documents through background processing
- **RESTful API**: Simple HTTP endpoints for file upload, status checking, and downloading

## Prerequisites

1. **Python 3.8+**: Required for running the application
2. **FFmpeg**: Required for audio processing
   - **macOS**: Install with Homebrew: `brew install ffmpeg`
   - **Windows**: Download from [FFmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **Linux**: Install with your package manager, e.g., `sudo apt install ffmpeg`
3. **OpenAI API Key**: Required for text-to-speech functionality (if using OpenAI's TTS)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd accuvio-manage-co2/infrastructure/pdf-to-podcast
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your Bedrock API credentials in app.py or using environment variables

## Usage

### Running the Service

Start the API server:

```bash
python app.py
```

The service will be available at http://localhost:8000

### API Endpoints

#### Convert PDF to Podcast

```
POST /convert
```

Upload a PDF file to convert it to podcast audio. For small files, returns the audio directly. For larger files, returns a job ID for status checking.

**Request:**
- Form data with "file" field containing the PDF file

**Response:**
- For small files: Audio file (MP3)
- For large files: JSON with job ID and status information

#### Check Conversion Status

```
GET /status/{job_id}
```

Check the status of a podcast conversion job.

**Response:**
- JSON with status information and download URL when complete

#### Download Podcast

```
GET /download/{job_id}
```

Download a completed podcast audio file.

**Response:**
- Audio file (MP3)

## Architecture

The service consists of three main components:

1. **FastAPI Web Service** (`app.py`): Handles HTTP requests, file uploads, and response delivery
2. **PDF to Podcast Service** (`service.py`): Core processing logic with three main steps:
   - Text extraction from PDF documents
   - Transformation to podcast script using AI
   - Text-to-speech conversion

## Configuration

Update the API configuration in `app.py`:

```python
service = PdfToPodcastService(
    bedrock_api_base="https://your-bedrock-endpoint.com",
    api_key="your-bedrock-api-key",
    tts_api_key="your-openai-api-key"  # Required for audio generation with OpenAI
)
```

For production use, it's recommended to use environment variables:

```python
import os
service = PdfToPodcastService(
    bedrock_api_base=os.getenv("BEDROCK_API_BASE"),
    api_key=os.getenv("BEDROCK_API_KEY"),
    tts_api_key=os.getenv("OPENAI_API_KEY")
)
```

## Dependencies

- FastAPI and Uvicorn for the web API
- PyPDF2 for PDF text extraction
- LangChain for AI integration
- Anthropic Claude 3.5 Sonnet model for text transformation
- Amazon Titan TTS for speech synthesis
- FFmpeg for audio processing (required for combining multiple audio files)

## Troubleshooting

### FFmpeg Not Found
If you see an error like `No such file or directory: 'ffmpeg'`:
1. Ensure FFmpeg is installed as per the prerequisites
2. Make sure the ffmpeg executable is in your system PATH
3. Restart the application after installation

If FFmpeg is not available, the application will still generate podcast scripts but may have issues combining audio files.

## License

[Specify your license here]
