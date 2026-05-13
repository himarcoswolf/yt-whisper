"""
Modelo Replicate: descarga audio de YouTube con yt-dlp + transcribe con Whisper Large v3.
Input: youtube_url (string)
Output: dict con title, duration, language, text, segments[]
"""
from cog import BasePredictor, Input
import yt_dlp
import whisper
import tempfile
import os


class Predictor(BasePredictor):
    def setup(self):
        """Carga el modelo Whisper Large v3 al arrancar el contenedor."""
        self.model = whisper.load_model("large-v3")

    def predict(
        self,
        youtube_url: str = Input(description="URL del vídeo de YouTube"),
        language: str = Input(
            default="es",
            description="Código de idioma (es, en, fr...) o 'auto' para detectar",
        ),
    ) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_template = os.path.join(tmpdir, "audio")

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_template + ".%(ext)s",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "128",
                    }
                ],
                "quiet": True,
                "no_warnings": True,
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    title = info.get("title", "")
                    duration = info.get("duration", 0)
                    author = info.get("uploader", "")
            except Exception as e:
                raise RuntimeError(f"yt-dlp falló al descargar el audio: {e}")

            mp3_path = audio_template + ".mp3"
            if not os.path.exists(mp3_path):
                for ext in ("webm", "m4a", "opus", "aac", "ogg"):
                    alt = f"{audio_template}.{ext}"
                    if os.path.exists(alt):
                        mp3_path = alt
                        break

            if not os.path.exists(mp3_path):
                raise RuntimeError("No se encontró el archivo de audio descargado por yt-dlp")

            lang = language if language and language != "auto" else None
            result = self.model.transcribe(
                mp3_path,
                language=lang,
                verbose=False,
                fp16=True,
            )

            return {
                "title": title,
                "duration": duration,
                "author": author,
                "language": result.get("language", language),
                "text": result["text"].strip(),
                "segments": [
                    {
                        "id": s.get("id"),
                        "start": s["start"],
                        "end": s["end"],
                        "text": s["text"].strip(),
                    }
                    for s in result["segments"]
                ],
            }
