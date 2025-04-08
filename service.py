import os
import PyPDF2
import asyncio
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import json
import base64
from tts_service import TextToSpeechService

class PdfToPodcastService:
    def __init__(self, bedrock_api_base, api_key, tts_api_key=None):
        """
        Initialize the PDF to Podcast service.
        
        Args:
            bedrock_api_base: Base URL of the Bedrock API endpoint
            api_key: API key for accessing Bedrock
            tts_api_key: API key for TTS service (optional)
        """
        self.bedrock_api_base = bedrock_api_base
        self.api_key = api_key
        self.tts_api_key = tts_api_key
        #self.tts_api_key = api_key
        self.tts_service = TextToSpeechService(api_key=tts_api_key)
        
    async def create_podcast(self, pdf_path, output_audio_path):
        """
        Process a PDF file and convert it to a podcast audio file.
        
        Args:
            pdf_path: Path to the PDF file
            output_audio_path: Path where the audio file will be saved
            
        Returns:
            Path to the generated audio file
        """
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        
        # Transform text to podcast script
        podcast_script = await self._transform_to_podcast(text)
        
        # Convert script to audio
        await self._generate_audio(podcast_script, output_audio_path)
        
        # Clean up the PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        return output_audio_path
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
        
        return text
    
    async def _transform_to_podcast(self, text):
        """
        Transform extracted text into a podcast script using Bedrock AI via LangChain
        """
        try:
            # Split the text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=500
            )
            chunks = text_splitter.split_text(text)
            
            # Initialize LLM
            llm = ChatOpenAI(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                temperature=0.5,
                openai_api_key=self.api_key,
                openai_api_base=self.bedrock_api_base,
            )
            
            # Updated template for a natural, conversational two-host podcast format
            template = """You are creating a podcast script for a natural-sounding live conversation between two expert hosts. The podcast is targeted at senior executives and board directors, but should sound like a genuine dialogue, not a scripted reading.

Your task is to transform the following content into an engaging conversation between Host A (David) and Host B (Sarah) with the following characteristics:

1. Natural dialogue patterns with realistic speech patterns, including occasional brief pauses, thoughtful considerations, and gentle interruptions where appropriate
2. Conversational transitions and authentic exchanges that show the hosts are actively listening to each other
3. Varied sentence structures and speech patterns unique to each host (David is concise and data-focused, Sarah provides strategic analysis and broader implications)
4. Include natural elements that would appear in actual conversation: brief agreements ("That's right", "I see"), clarifying questions, building on the other's points
5. Incorporate realistic verbal cues like "you know," "I think," "as you mentioned," that make dialogue sound authentic

The tone should be authoritative and professional, but conversational and natural - avoid overly formal language that sounds stilted when spoken aloud. The hosts should sound knowledgeable but approachable.

IMPORTANT: Avoid scripted-sounding phrases like "[Pause]" or "[thoughtful tone]" - instead, use natural dialogue patterns that convey these elements implicitly.

PDF CONTENT:
{text}

PODCAST CONVERSATION:"""
            
            prompt = PromptTemplate.from_template(template)
            chain = LLMChain(prompt=prompt, llm=llm)
            
            # Process each chunk and combine results
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    # First chunk gets the introduction
                    result = await chain.ainvoke({"text": chunk})
                    processed_chunks.append(result["text"])
                elif i == len(chunks) - 1:
                    # Updated conclusion template for natural conversation
                    conclude_template = """Continue and conclude the conversation between David and Sarah with this final section:
                    
{text}

Have them naturally summarize key takeaways and strategic implications as they would in a live discussion. End with a genuine-sounding sign-off that feels unscripted and authentic."""
                    
                    conclude_prompt = PromptTemplate.from_template(conclude_template)
                    conclude_chain = LLMChain(prompt=conclude_prompt, llm=llm)
                    result = await conclude_chain.ainvoke({"text": chunk})
                    processed_chunks.append(result["text"])
                else:
                    # Updated middle section template for natural conversation
                    continue_template = """Continue the natural conversation between David and Sarah discussing the following content:
                    
{text}

Maintain the authentic dialogue feel with natural speech patterns. Have them build on each other's points and demonstrate active listening. Keep the focus on strategic implications and executive-level insights while sounding like a genuine conversation."""
                    
                    continue_prompt = PromptTemplate.from_template(continue_template)
                    continue_chain = LLMChain(prompt=continue_prompt, llm=llm)
                    result = await continue_chain.ainvoke({"text": chunk})
                    processed_chunks.append(result["text"])
            
            # Join all processed chunks
            return "\n\n".join(processed_chunks)
        
        except Exception as e:
            raise Exception(f"Failed to transform text to podcast format: {str(e)}")
    
    async def _generate_audio(self, text, output_path):
        """
        Convert the podcast script to audio using TTS service
        """
        try:
            # First, save the text as a fallback
            text_output_path = output_path.replace('.mp3', '.txt')
            with open(text_output_path, 'w') as f:
                f.write(text)
            
            print(f"Saved podcast script text to {text_output_path}")
            
            # Check if TTS API key is available before attempting audio generation
            if self.tts_api_key:
                print("Attempting to generate audio using external TTS service...")
                try:
                    return await self.tts_service.generate_audio(text, output_path, voice="alloy")
                except Exception as tts_error:
                    print(f"TTS generation failed: {str(tts_error)}")
                    # Return the text file path instead of the non-existent mp3
                    return text_output_path
            else:
                print("No TTS API key provided, skipping audio generation")
                return text_output_path
            
        except Exception as e:
            print(f"Failed to generate audio: {str(e)}")
            return text_output_path
    
    def _split_for_tts(self, text, max_length):
        """Split text into smaller chunks suitable for TTS processing"""
        # Split by paragraphs first (to maintain natural breaks)
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds max length, start a new chunk
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
        """Combine multiple MP3 files into a single file"""
        try:
            import subprocess
            
            # Check if ffmpeg is installed
            try:
                # Try to execute ffmpeg to see if it's available
                subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("FFmpeg not found. Please install FFmpeg to enable audio combination.")
                # Fall back to using just the first audio chunk
                if audio_files and os.path.exists(audio_files[0]):
                    import shutil
                    shutil.copy(audio_files[0], output_path)
                    return
                else:
                    raise Exception("FFmpeg not found and no audio files available")
            
            # Create a file list for ffmpeg
            with open("filelist.txt", "w") as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            # Use ffmpeg to concatenate the files
            subprocess.run([
                "ffmpeg", "-f", "concat", "-safe", "0", 
                "-i", "filelist.txt", "-c", "copy", output_path
            ], check=True)
            
            # Clean up
            if os.path.exists("filelist.txt"):
                os.remove("filelist.txt")
                
        except Exception as e:
            print(f"Error combining audio files: {str(e)}")
            # Fall back to using just the first audio chunk if available
            if audio_files and os.path.exists(audio_files[0]):
                import shutil
                shutil.copy(audio_files[0], output_path)
            else:
                raise Exception(f"Failed to combine audio files: {str(e)}")

