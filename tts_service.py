import os
import requests
import json
from pathlib import Path
import tempfile
import subprocess

class TextToSpeechService:
    """Service to convert text to speech using various providers"""
    
    def __init__(self, api_key=None, service="openai"):
        """
        Initialize the TTS service.
        
        Args:
            api_key: API key for the TTS service
            service: Service provider (openai, elevenlabs, etc)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.service = service
        
    async def generate_audio(self, text, output_path, voice="alloy"):
        """
        Convert text to speech and save to file.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            voice: Voice ID to use (model-specific)
            
        Returns:
            Path to the generated audio file
        """
        if self.service == "openai":
            return await self._generate_audio_openai(text, output_path, voice)
        elif self.service == "local":
            return self._generate_audio_local(text, output_path)
        else:
            raise ValueError(f"Unsupported TTS service: {self.service}")
    
    async def _generate_audio_openai(self, text, output_path, voice="alloy"):
        """
        Use OpenAI's TTS API with specified voice
        
        Available voices:
        - alloy: Neutral voice
        - echo: Male voice
        - fable: Male voice with a warm tone
        - onyx: Male voice with gravitas
        - nova: Female voice
        - shimmer: Female voice with a clear, bright tone
        """
        try:
            # OpenAI has a limit on input length, so we need to split the text
            max_chunk_size = 4000
            chunks = self._split_text_for_tts(text, max_chunk_size)
            temp_files = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "tts-1",
                    "input": chunk,
                    "voice": voice,  # Will use the specified voice (echo for male, nova for female)
                    "response_format": "mp3"
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    # Save chunk to temporary file
                    temp_file = f"{output_path}_chunk_{i}.mp3"
                    with open(temp_file, "wb") as f:
                        f.write(response.content)
                    temp_files.append(temp_file)
                else:
                    print(f"Error: {response.status_code}, {response.text}")
                    # Continue processing other chunks
            
            # Combine chunks if multiple
            if len(temp_files) > 1:
                self._combine_audio_files(temp_files, output_path)
                # Clean up temp files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            elif len(temp_files) == 1:
                # Just rename the single file
                os.rename(temp_files[0], output_path)
            else:
                # No audio generated, create a text file instead
                text_path = output_path.replace('.mp3', '.txt')
                with open(text_path, 'w') as f:
                    f.write(text)
                return text_path
                
            return output_path
            
        except Exception as e:
            print(f"Failed to generate audio with OpenAI: {str(e)}")
            # Fall back to saving as text
            text_path = output_path.replace('.mp3', '.txt')
            with open(text_path, 'w') as f:
                f.write(text)
            return text_path
    
    def _generate_audio_local(self, text, output_path):
        """Use local TTS system if available (fallback)"""
        # Try using system TTS if available (Mac OS)
        try:
            # Save text to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
                temp_name = temp.name
                temp.write(text.encode('utf-8'))
            
            # Use say command (Mac) or alternative
            if os.name == 'posix':  # Mac or Linux
                if os.path.exists('/usr/bin/say'):  # Mac
                    subprocess.run(['say', '-f', temp_name, '-o', output_path], check=True)
                    return output_path
            
            # If we get here, local TTS failed
            text_path = output_path.replace('.mp3', '.txt')
            with open(text_path, 'w') as f:
                f.write(text)
            return text_path
            
        except Exception as e:
            print(f"Local TTS failed: {str(e)}")
            # Fall back to text file
            text_path = output_path.replace('.mp3', '.txt')
            with open(text_path, 'w') as f:
                f.write(text)
            return text_path
        finally:
            # Clean up temp file
            if 'temp_name' in locals() and os.path.exists(temp_name):
                os.remove(temp_name)
    
    def _split_text_for_tts(self, text, max_length):
        """Split text into manageable chunks for TTS processing"""
        # Split by paragraph to maintain natural breaks
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding paragraph exceeds max length, start new chunk
            if len(current_chunk) + len(paragraph) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _combine_audio_files(self, audio_files, output_path):
        """Combine multiple audio files into one"""
        try:
            # Create a file list for ffmpeg
            with open("filelist.txt", "w") as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            # Use ffmpeg to concatenate files
            subprocess.run([
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", "filelist.txt", "-c", "copy", output_path
            ], check=True)
            
            # Clean up
            os.remove("filelist.txt")
        except Exception as e:
            print(f"Error combining audio files: {str(e)}")
            # If combining fails, just use the first file
            if audio_files and os.path.exists(audio_files[0]):
                os.rename(audio_files[0], output_path)
