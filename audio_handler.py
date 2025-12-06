import os
from openai import OpenAI
from dotenv import load_dotenv
import tempfile

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

class AudioHandler:
    def __init__(self):
        # Initialize OpenAI Client
        # You can swap this for Anthropic/Google if using different providers, 
        # but OpenAI handles Audio best currently.
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe_audio(self, audio_bytes):
        """
        ASR: Converts Audio Bytes (from microphone) to Text using Whisper.
        """
        try:
            # Create a temporary file to save the recorded bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # Call OpenAI Whisper API
            with open(temp_audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            
            # Cleanup temp file
            os.remove(temp_audio_path)
            
            return transcription.text

        except Exception as e:
            return f"Error in transcription: {str(e)}"

    def text_to_speech(self, text):
        """
        TTS: Converts Text to Audio File using OpenAI TTS.
        Returns the path to the generated audio file.
        """
        try:
            # Limit text length if necessary (optional cost saving)
            if not text:
                return None

            response = self.client.audio.speech.create(
                model="tts-1", # or "tts-1-hd" for higher quality
                voice="alloy", # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text
            )

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
                response.stream_to_file(temp_tts.name)
                return temp_tts.name

        except Exception as e:
            print(f"Error in TTS: {str(e)}")
            return None