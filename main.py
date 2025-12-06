import os
import pathlib
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audio_handler import AudioHandler


# ===================================================
# Initialize FastAPI and Audio Engine
# ===================================================
app = FastAPI(title="VoiceShop Audio API")
audio_handler = AudioHandler()


# ===================================================
# CORS Settings — allow frontend to call backend
# ===================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================
# Serve React Frontend (dist/)
# ===================================================
BASE_DIR = pathlib.Path(__file__).parent
DIST_DIR = BASE_DIR / "dist"

if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
    # Mount the static frontend at root "/"
    app.mount(
        "/",
        StaticFiles(directory=str(DIST_DIR), html=True),
        name="static",
    )
else:
    print("⚠ WARNING: dist/ folder not found — frontend will NOT be served.")


# ===================================================
# Pydantic Models
# ===================================================
class TTSRequest(BaseModel):
    text: str


# ===================================================
# Health Check
# ===================================================
@app.get("/health")
async def health():
    """Simple endpoint to verify the server is running."""
    return {"status": "ok"}


# ===================================================
# Speech-to-Text API (Whisper)
# ===================================================
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Accepts a microphone audio Blob from the frontend,
    sends it to Whisper for transcription,
    and returns the recognized text.
    """
    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty.")

        text = audio_handler.transcribe_audio(audio_bytes)

        if text.startswith("Error"):
            raise HTTPException(status_code=500, detail=text)

        return {"text": text}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ===================================================
# Text-to-Speech API (OpenAI TTS)
# ===================================================
@app.post("/api/tts")
async def tts(payload: TTSRequest, background_tasks: BackgroundTasks):
    """
    Accepts a text string and returns an MP3 file
    generated using OpenAI's text-to-speech.
    """
    try:
        text = payload.text.strip()

        if not text:
            raise HTTPException(status_code=400, detail="Text is empty.")

        audio_path = audio_handler.text_to_speech(text)

        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="TTS failed.")

        # Schedule deletion of temp audio file
        background_tasks.add_task(os.remove, audio_path)

        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="tts.mp3"
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
