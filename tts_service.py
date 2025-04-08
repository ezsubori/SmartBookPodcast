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
        Use OpenAI's TTS API with specified voice and improved naturalness
        
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
                
                # Use TTS-1-HD for higher quality, more natural speech
                model = "tts-1-hd"
                
                # Enhance the text with vocal dynamics
                enhanced_text = self._enhance_speech_dynamics(chunk, voice)
                
                # Adjust voice settings for more expressiveness
                # The voice_settings parameter isn't yet available in the API,
                # so we modify the text itself to achieve similar effects
                
                payload = {
                    "model": model,
                    "input": enhanced_text,
                    "voice": voice,
                    "response_format": "mp3",
                    # Speed affects pitch somewhat - slightly slower for deeper voices,
                    # faster for higher pitch variation
                    "speed": 0.92 if voice in ["onyx"] else 1.08 if voice == "nova" else 1.0
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
    
    def _enhance_natural_speech(self, text, voice):
        """Add natural speech elements to improve fluency"""
        # For British accent (echo voice), add some British speech patterns
        if voice == "echo":
            # Add slight pauses with commas
            text = text.replace(". ", ", right. ")
            text = text.replace("? ", "? Well, ")
            
            # Add British filler words and expressions
            text = text.replace("I think ", "I rather think ")
            text = text.replace("That's good", "That's quite good")
            
            # Add SSML for more control if using a service that supports it
            # Since OpenAI doesn't support SSML yet, we'll use text modifications
            
        # For female voice (nova), add different speech patterns
        elif voice == "nova":
            # Different filler patterns for Sarah
            text = text.replace("I think", "I believe")
            text = text.replace("good point", "excellent point")
        
        return text

    def _enhance_speech_dynamics(self, text, voice):
        """
        Enhance text to create more natural speech patterns and pitch variation
        """
        # Add punctuation and formatting that will create more dynamic speech
        
        # Add question marks to create rising intonation
        text = text.replace(", right.", ", right?")
        text = text.replace(", correct.", ", correct?")
        
        # Use exclamation points sparingly for emphasis
        text = text.replace(" significant increase", " significant increase!")
        text = text.replace(" remarkable growth", " remarkable growth!")
        
        # Add commas and ellipses for pacing and emphasis
        text = text.replace(". And", ". ...And")
        text = text.replace(". But", ". ...But")
        
        # Create more natural pauses with commas
        text = text.replace(" however ", ", however, ")
        text = text.replace(" therefore ", ", therefore, ")
        
        # Add emphasis markers that affect pitch in TTS
        if "increased by" in text:
            text = text.replace("increased by", "*increased* by")
        
        if "percent" in text:
            text = text.replace("percent", "*percent*")
        
        # Voice-specific enhancements
        if voice == "onyx":  # David's British voice
            # Add British expressions and emphasis patterns
            text = text.replace("That's right", "That's absolutely right")
            text = text.replace("very good", "*very* good")
            text = text.replace("I agree", "I quite agree")
            text = text.replace("looking at", "looking *carefully* at")
            # Add pauses where a British speaker might pause
            text = text.replace(". What", ". ...What")
            text = text.replace(". Now,", ". ...Now,")
        
        elif voice == "nova":  # Sarah's voice
            # Add patterns that create higher pitch variation
            text = text.replace("interesting", "*really* interesting")
            text = text.replace("important", "*critically* important")
            text = text.replace("strategic", "*strategic*")
            # Create questioning intonation in analysis
            text = text.replace("we should consider", "shouldn't we consider")
            text = text.replace("this suggests", "this suggests, don't you think")
            # Add vocal fry markers at ends of statements
            text = text.replace(". The", "... The")
        
        # Remove any double spaces or repeated punctuation
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.replace("?..", "?.")
        text = text.replace("!..", "!.")
        text = text.replace("...", "...")
        
        return text
