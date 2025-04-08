from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import uuid
from service import PdfToPodcastService

app = FastAPI(title="PDF to Podcast API", description="Convert PDF documents to podcast audio")

# Create temp folders if they don't exist
os.makedirs("temp/uploads", exist_ok=True)
os.makedirs("temp/podcasts", exist_ok=True)

# Initialize service with Bedrock API endpoint
service = PdfToPodcastService(
    bedrock_api_base="https://hackfest-bedrock-proxy.diligentoneplatform-dev.com/api/v1",
    api_key="2ega5d1b34c4258c4b03e6c49ae3f9e1"  # Replace with actual API key or use environment variable
)

@app.post("/convert")
async def convert_pdf_to_podcast(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Converts a PDF document to a podcast audio file.
    
    Args:
        file: The PDF file to be converted
        
    Returns:
        Audio file of the podcast or job ID for status checking
    """
    # Validate the file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Generate unique ID for this conversion job
    job_id = str(uuid.uuid4())
    pdf_path = f"temp/uploads/{job_id}.pdf"
    audio_path = f"temp/podcasts/{job_id}.mp3"
    
    # Save the uploaded PDF
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # If the PDF is small, process immediately, otherwise do it in background
    try:
        file_size = os.path.getsize(pdf_path)
        
        if file_size < 1024 * 1024:  # If less than 1MB, process immediately
            await service.create_podcast(pdf_path, audio_path)
            return FileResponse(
                audio_path, 
                media_type="audio/mpeg",
                filename="podcast.mp3"
            )
        else:
            # For larger files, process in background
            if background_tasks:
                background_tasks.add_task(service.create_podcast, pdf_path, audio_path)
            else:
                # Start processing in separate thread/process
                import asyncio
                asyncio.create_task(service.create_podcast(pdf_path, audio_path))
                
            return JSONResponse({
                "status": "processing",
                "job_id": job_id,
                "message": "Your podcast is being generated. Check status at /status/{job_id}"
            })
    except Exception as e:
        # Clean up on error
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    """Check the status of a podcast conversion job"""
    audio_path = f"temp/podcasts/{job_id}.mp3"
    
    if os.path.exists(audio_path):
        return {
            "status": "completed",
            "download_url": f"/download/{job_id}"
        }
    
    pdf_path = f"temp/uploads/{job_id}.pdf"
    if os.path.exists(pdf_path):
        return {
            "status": "processing",
            "message": "Your podcast is still being generated"
        }
    
    raise HTTPException(status_code=404, detail="Job not found")

@app.get("/download/{job_id}")
async def download_podcast(job_id: str):
    """Download a completed podcast"""
    audio_path = f"temp/podcasts/{job_id}.mp3"
    
    if os.path.exists(audio_path):
        return FileResponse(
            audio_path, 
            media_type="audio/mpeg",
            filename="podcast.mp3"
        )
    
    raise HTTPException(status_code=404, detail="Podcast not found")

@app.on_event("shutdown")
def cleanup():
    """Clean up temporary files on shutdown"""
    import shutil
    try:
        shutil.rmtree("temp/uploads")
        shutil.rmtree("temp/podcasts")
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
