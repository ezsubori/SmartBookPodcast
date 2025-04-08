    
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

class PdfToPodcastService:
    def __init__(self, bedrock_api_base, api_key):
        """
        Initialize the PDF to Podcast service.
        
        Args:
            bedrock_api_base: Base URL of the Bedrock API endpoint
            api_key: API key for accessing Bedrock
        """
        self.bedrock_api_base = bedrock_api_base
        self.api_key = api_key
        
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
            
            # Prepare template for podcast transformation
            template = """You are an expert podcaster who creates engaging, conversational content.

Your task is to transform the following PDF content into a natural-sounding podcast script. 

The script should:
1. Begin with a warm welcome and brief introduction
2. Present the content in a conversational, engaging tone
3. Break complex concepts into digestible segments
4. Include occasional rhetorical questions or hooks to maintain listener interest
5. End with a summary and conclusion

Remember, this should sound like people talking, not like someone reading an article.

PDF CONTENT:
{text}

PODCAST SCRIPT:"""
            
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
                    # Last chunk gets the conclusion
                    conclude_template = """Continue the podcast and provide a thoughtful conclusion for this content:
                    
{text}

Conclude the podcast with a summary of key points and a friendly sign-off."""
                    
                    conclude_prompt = PromptTemplate.from_template(conclude_template)
                    conclude_chain = LLMChain(prompt=conclude_prompt, llm=llm)
                    result = await conclude_chain.ainvoke({"text": chunk})
                    processed_chunks.append(result["text"])
                else:
                    # Middle chunks continue the narrative
                    continue_template = """Continue the podcast with the following content:
                    
{text}

Continue the conversational podcast tone without any introduction or conclusion."""
                    
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
        Convert the podcast script to audio using Bedrock AI text-to-speech
        """
        try:
            # Split text into chunks if it's too long for TTS
            max_tts_length = 3000  # Adjust based on API limits
            text_chunks = self._split_for_tts(text, max_tts_length)
            
            audio_chunks = []
            
            for chunk in text_chunks:
                # Call Bedrock API for TTS
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "model": "amazon.titan-tts-expressive",
                    "input": chunk,
                    "voice_preset": "female_casual", # Options like female_formal, male_casual, etc.
                    "response_format": "mp3"
                }
                
                response = requests.post(
                    f"{self.bedrock_api_base}/audio/speech",
                    headers=headers,
                    data=json.dumps(payload)
                )
                
                if response.status_code != 200:
                    raise Exception(f"TTS API error: {response.status_code}, {response.text}")
                
                # Save audio chunk
                temp_chunk_path = f"{output_path}_chunk_{len(audio_chunks)}.mp3"
                with open(temp_chunk_path, "wb") as f:
                    f.write(response.content)
                
                audio_chunks.append(temp_chunk_path)
            
            # Combine audio chunks if needed
            if len(audio_chunks) > 1:
                self._combine_audio_files(audio_chunks, output_path)
                # Clean up temporary files
                for chunk_path in audio_chunks:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            elif len(audio_chunks) == 1:
                # Rename single chunk to final output name
                os.rename(audio_chunks[0], output_path)
                
        except Exception as e:
            raise Exception(f"Failed to generate audio: {str(e)}")
    
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
        # This is a simplified version - in a production environment,
        # you would likely use a library like pydub to properly combine the audio files
        import subprocess
        
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
        os.remove("filelist.txt")



# import os
# import PyPDF2
# import asyncio
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# import requests
# import json
# import base64

# class PdfToPodcastService:
#     def __init__(self, bedrock_api_base, api_key):
#         """
#         Initialize the PDF to Podcast service.
        
#         Args:
#             bedrock_api_base: Base URL of the Bedrock API endpoint
#             api_key: API key for accessing Bedrock
#         """
#         self.bedrock_api_base = bedrock_api_base
#         self.api_key = api_key
        
#     async def create_podcast(self, pdf_path, output_audio_path):
#         """
#         Process a PDF file and convert it to a podcast audio file.
        
#         Args:
#             pdf_path: Path to the PDF file
#             output_audio_path: Path where the audio file will be saved
            
#         Returns:
#             Path to the generated audio file
#         """
#         # Extract text from PDF
#         text = self._extract_text_from_pdf(pdf_path)
        
#         # Transform text to podcast script
#         podcast_script = await self._transform_to_podcast(text)
        
#         # Convert script to audio
#         await self._generate_audio(podcast_script, output_audio_path)
        
#         # Clean up the PDF file
#         if os.path.exists(pdf_path):
#             os.remove(pdf_path)
        
#         return output_audio_path
    
#     def _extract_text_from_pdf(self, pdf_path):
#         """Extract text content from a PDF file"""
#         text = ""
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 for page in reader.pages:
#                     text += page.extract_text() + "\n\n"
#         except Exception as e:
#             raise Exception(f"Failed to extract text from PDF: {str(e)}")
        
#         return text
    
#     async def _transform_to_podcast(self, text):
#         """
#         Transform extracted text into a podcast script using Bedrock AI via LangChain
#         """
#         try:
#             # Split the text into manageable chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=10000,
#                 chunk_overlap=500
#             )
#             chunks = text_splitter.split_text(text)
            
#             # Initialize LLM

#             from langchain_openai import ChatOpenAI

#             llm = ChatOpenAI(
#                 model="anthropic.claude-3-5-sonnet-20241022-v2:0",
#                 temperature=0.5,
#                 api_key=self.api_key,
#                 base_url=self.bedrock_api_base,
#             )
            
#             # Prepare template for podcast transformation
#             template = """You are an expert podcaster who creates engaging, conversational content.

# Your task is to transform the following PDF content into a natural-sounding podcast script. 

# The script should:
# 1. Begin with a warm welcome and brief introduction
# 2. Present the content in a conversational, engaging tone
# 3. Break complex concepts into digestible segments
# 4. Include occasional rhetorical questions or hooks to maintain listener interest
# 5. End with a summary and conclusion

# Remember, this should sound like people talking, not like someone reading an article.

# PDF CONTENT:
# {text}

# PODCAST SCRIPT:"""
            
#             prompt = PromptTemplate.from_template(template)
#             chain = LLMChain(prompt=prompt, llm=llm)
#             # from langchain.chains import create_extraction_chain
#             # chain = LLMChain.from_string(llm=llm, template=template, output_key="text")
            
#             # Process each chunk and combine results
#             processed_chunks = []
            
#             for i, chunk in enumerate(chunks):
#                 if i == 0:
#                     # First chunk gets the introduction
#                     result = await chain.ainvoke({"text": chunk})
#                     processed_chunks.append(result["text"])
#                 elif i == len(chunks) - 1:
#                     # Last chunk gets the conclusion
#                     conclude_template = """Continue the podcast and provide a thoughtful conclusion for this content:
                    
# {text}

# Conclude the podcast with a summary of key points and a friendly sign-off."""
                    
#                     conclude_prompt = PromptTemplate.from_template(conclude_template)
#                     conclude_chain = LLMChain(prompt=conclude_prompt, llm=llm)
#                     result = await conclude_chain.ainvoke({"text": chunk})
#                     processed_chunks.append(result["text"])
#                 else:
#                     # Middle chunks continue the narrative
#                     continue_template = """Continue the podcast with the following content:
                    
# {text}

# Continue the conversational podcast tone without any introduction or conclusion."""
                    
#                     continue_prompt = PromptTemplate.from_template(continue_template)
#                     continue_chain = LLMChain(prompt=continue_prompt, llm=llm)
#                     result = await continue_chain.ainvoke({"text": chunk})
#                     processed_chunks.append(result["text"])
            
#             # Join all processed chunks
#             return "\n\n".join(processed_chunks)
        
#         except Exception as e:
#             raise Exception(f"Failed to transform text to podcast format: {str(e)}")
    
#     async def _generate_audio(self, text, output_path):
#         """
#         Convert the podcast script to audio using Bedrock AI text-to-speech
#         """
#         try:
#             # Split text into chunks if it's too long for TTS
#             max_tts_length = 3000  # Adjust based on API limits
#             text_chunks = self._split_for_tts(text, max_tts_length)
            
#             audio_chunks = []
            
#             for chunk in text_chunks:
#                 # Call Bedrock API for TTS
#                 headers = {
#                     "Content-Type": "application/json",
#                     "Authorization": f"Bearer {self.api_key}"
#                 }
                
#                 payload = {
#                     "model": "amazon.titan-tts-expressive",
#                     "input": chunk,
#                     "voice_preset": "female_casual", # Options like female_formal, male_casual, etc.
#                     "response_format": "mp3"
#                 }
                
#                 response = requests.post(
#                     f"{self.bedrock_api_base}/audio/speech",
#                     headers=headers,
#                     data=json.dumps(payload)
#                 )
                
#                 if response.status_code != 200:
#                     raise Exception(f"TTS API error: {response.status_code}, {response.text}")
                
#                 # Save audio chunk
#                 temp_chunk_path = f"{output_path}_chunk_{len(audio_chunks)}.mp3"
#                 with open(temp_chunk_path, "wb") as f:
#                     f.write(response.content)
                
#                 audio_chunks.append(temp_chunk_path)
            
#             # Combine audio chunks if needed
#             if len(audio_chunks) > 1:
#                 self._combine_audio_files(audio_chunks, output_path)
#                 # Clean up temporary files
#                 for chunk_path in audio_chunks:
#                     if os.path.exists(chunk_path):
#                         os.remove(chunk_path)
#             elif len(audio_chunks) == 1:
#                 # Rename single chunk to final output name
#                 os.rename(audio_chunks[0], output_path)
                
#         except Exception as e:
#             raise Exception(f"Failed to generate audio: {str(e)}")
    
#     def _split_for_tts(self, text, max_length):
#         """Split text into smaller chunks suitable for TTS processing"""
#         # Split by paragraphs first (to maintain natural breaks)
#         paragraphs = text.split('\n')
#         chunks = []
#         current_chunk = ""
        
#         for paragraph in paragraphs:
#             # If adding this paragraph exceeds max length, start a new chunk
#             if len(current_chunk) + len(paragraph) > max_length:
#                 if current_chunk:
#                     chunks.append(current_chunk)
#                 current_chunk = paragraph
#             else:
#                 # Add paragraph to current chunk
#                 if current_chunk:
#                     current_chunk += "\n" + paragraph
#                 else:
#                     current_chunk = paragraph
        
#         # Add the last chunk if not empty
#         if current_chunk:
#             chunks.append(current_chunk)
            
#         return chunks
    
#     def _combine_audio_files(self, audio_files, output_path):
#         """Combine multiple MP3 files into a single file"""
#         # This is a simplified version - in a production environment,
#         # you would likely use a library like pydub to properly combine the audio files
#         import subprocess
        
#         # Create a file list for ffmpeg
#         with open("filelist.txt", "w") as f:
#             for audio_file in audio_files:
#                 f.write(f"file '{audio_file}'\n")
        
#         # Use ffmpeg to concatenate the files
#         subprocess.run([
#             "ffmpeg", "-f", "concat", "-safe", "0", 
#             "-i", "filelist.txt", "-c", "copy", output_path
#         ], check=True)
        
#         # Clean up
#         os.remove("filelist.txt")

# import os
# import PyPDF2
# import asyncio
# from langchain.chains.llm import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_text_splitters import RecursiveCharacterTextSplitter
 
# import requests
# import json
# import base64
 
# class PdfToPodcastService:
#     def __init__(self, bedrock_api_base, api_key):
#         """
#         Initialize the PDF to Podcast service.
        
#         Args:
#             bedrock_api_base: Base URL of the Bedrock API endpoint
#             api_key: API key for accessing Bedrock
#         """
#         self.bedrock_api_base = bedrock_api_base
#         self.api_key = api_key
        
#     async def create_podcast(self, pdf_path, output_audio_path):
#         """
#         Process a PDF file and convert it to a podcast audio file.
        
#         Args:
#             pdf_path: Path to the PDF file
#             output_audio_path: Path where the audio file will be saved
            
#         Returns:
#             Path to the generated audio file
#         """
#         # Extract text from PDF
#         text = self._extract_text_from_pdf(pdf_path)
        
#         # Transform text to podcast script
#         podcast_script = await self._transform_to_podcast(text)
        
#         # Convert script to audio
#         await self._generate_audio(podcast_script, output_audio_path)
        
#         # Clean up the PDF file
#         if os.path.exists(pdf_path):
#             os.remove(pdf_path)
        
#         return output_audio_path
    
#     def _extract_text_from_pdf(self, pdf_path):
#         """Extract text content from a PDF file"""
#         text = ""
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 for page in reader.pages:
#                     text += page.extract_text() + "\n\n"
#         except Exception as e:
#             raise Exception(f"Failed to extract text from PDF: {str(e)}")
        
#         return text
    
#     async def _transform_to_podcast(self, text):
#         """
#         Transform extracted text into a podcast script using Bedrock AI via LangChain
#         """
#         try:
#             # Split the text into manageable chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=10000,
#                 chunk_overlap=500
#             )
#             chunks = text_splitter.split_text(text)
            
#             # Initialize LLM
 
#             from langchain_core.language_models.chat_models import ChatOpenAI
 
#             llm = ChatOpenAI(
#                 model="anthropic.claude-3-5-sonnet-20241022-v2:0",
#                 temperature=0.5,
#                 api_key=self.api_key,
#                 base_url=self.bedrock_api_base,
#             )
            
#             # Prepare template for podcast transformation
#             template = """You are an expert podcaster who creates engaging, conversational content.
 
# Your task is to transform the following PDF content into a natural-sounding podcast script.
 
# The script should:
# 1. Begin with a warm welcome and brief introduction
# 2. Present the content in a conversational, engaging tone
# 3. Break complex concepts into digestible segments
# 4. Include occasional rhetorical questions or hooks to maintain listener interest
# 5. End with a summary and conclusion
 
# Remember, this should sound like people talking, not like someone reading an article.
 
# PDF CONTENT:
# {text}
 
# PODCAST SCRIPT:"""
            
#             prompt = PromptTemplate.from_template(template)
#             chain = LLMChain(prompt=prompt, llm=llm)
            
#             # Process each chunk and combine results
#             processed_chunks = []
            
#             for i, chunk in enumerate(chunks):
#                 if i == 0:
#                     # First chunk gets the introduction
#                     result = await chain.ainvoke({"text": chunk})
#                     processed_chunks.append(result["text"])
#                 elif i == len(chunks) - 1:
#                     # Last chunk gets the conclusion
#                     conclude_template = """Continue the podcast and provide a thoughtful conclusion for this content:
                    
# {text}
 
# Conclude the podcast with a summary of key points and a friendly sign-off."""
                    
#                     conclude_prompt = PromptTemplate.from_template(conclude_template)
#                     conclude_chain = LLMChain(prompt=conclude_prompt, llm=llm)
#                     result = await conclude_chain.ainvoke({"text": chunk})
#                     processed_chunks.append(result["text"])
#                 else:
#                     # Middle chunks continue the narrative
#                     continue_template = """Continue the podcast with the following content:
                    
# {text}
 
# Continue the conversational podcast tone without any introduction or conclusion."""
                    
#                     continue_prompt = PromptTemplate.from_template(continue_template)
#                     continue_chain = LLMChain(prompt=continue_prompt, llm=llm)
#                     result = await continue_chain.ainvoke({"text": chunk})
#                     processed_chunks.append(result["text"])
            
#             # Join all processed chunks
#             return "\n\n".join(processed_chunks)
        
#         except Exception as e:
#             raise Exception(f"Failed to transform text to podcast format: {str(e)}")
    
#     async def _generate_audio(self, text, output_path):
#         """
#         Convert the podcast script to audio using Bedrock AI text-to-speech
#         """
#         try:
#             # Split text into chunks if it's too long for TTS
#             max_tts_length = 3000  # Adjust based on API limits
#             text_chunks = self._split_for_tts(text, max_tts_length)
            
#             audio_chunks = []
            
#             for chunk in text_chunks:
#                 # Call Bedrock API for TTS
#                 headers = {
#                     "Content-Type": "application/json",
#                     "Authorization": f"Bearer {self.api_key}"
#                 }
                
#                 payload = {
#                     "model": "amazon.titan-tts-expressive",
#                     "input": chunk,
#                     "voice_preset": "female_casual", # Options like female_formal, male_casual, etc.
#                     "response_format": "mp3"
#                 }
                
#                 response = requests.post(
#                     f"{self.bedrock_api_base}/audio/speech",
#                     headers=headers,
#                     data=json.dumps(payload)
#                 )
                
#                 if response.status_code != 200:
#                     raise Exception(f"TTS API error: {response.status_code}, {response.text}")
                
#                 # Save audio chunk
#                 temp_chunk_path = f"{output_path}_chunk_{len(audio_chunks)}.mp3"
#                 with open(temp_chunk_path, "wb") as f:
#                     f.write(response.content)
                
#                 audio_chunks.append(temp_chunk_path)
            
#             # Combine audio chunks if needed
#             if len(audio_chunks) > 1:
#                 self._combine_audio_files(audio_chunks, output_path)
#                 # Clean up temporary files
#                 for chunk_path in audio_chunks:
#                     if os.path.exists(chunk_path):
#                         os.remove(chunk_path)
#             elif len(audio_chunks) == 1:
#                 # Rename single chunk to final output name
#                 os.rename(audio_chunks[0], output_path)
                
#         except Exception as e:
#             raise Exception(f"Failed to generate audio: {str(e)}")
    
#     def _split_for_tts(self, text, max_length):
#         """Split text into smaller chunks suitable for TTS processing"""
#         # Split by paragraphs first (to maintain natural breaks)
#         paragraphs = text.split('\n')
#         chunks = []
#         current_chunk = ""
        
#         for paragraph in paragraphs:
#             # If adding this paragraph exceeds max length, start a new chunk
#             if len(current_chunk) + len(paragraph) > max_length:
#                 if current_chunk:
#                     chunks.append(current_chunk)
#                 current_chunk = paragraph
#             else:
#                 # Add paragraph to current chunk
#                 if current_chunk:
#                     current_chunk += "\n" + paragraph
#                 else:
#                     current_chunk = paragraph
        
#         # Add the last chunk if not empty
#         if current_chunk:
#             chunks.append(current_chunk)
            
#         return chunks
    
#     def _combine_audio_files(self, audio_files, output_path):
#         """Combine multiple MP3 files into a single file"""
#         # This is a simplified version - in a production environment,
#         # you would likely use a library like pydub to properly combine the audio files
#         import subprocess
        
#         # Create a file list for ffmpeg
#         with open("filelist.txt", "w") as f:
#             for audio_file in audio_files:
#                 f.write(f"file '{audio_file}'\n")
        
#         # Use ffmpeg to concatenate the files
#         subprocess.run([
#             "ffmpeg", "-f", "concat", "-safe", "0", 
#             "-i", "filelist.txt", "-c", "copy", output_path
#         ], check=True)
        
#         # Clean up
#         os.remove("filelist.txt")
 
 