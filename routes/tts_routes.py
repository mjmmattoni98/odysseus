# routes/tts_routes.py
"""
TTS API routes — multi-provider (local Kokoro, API endpoint, browser).
"""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel
import logging

from services.tts.tts_service import DEFAULT_KOKORO_VOICE, kokoro_voice_groups

logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    format: str = "audio"  # "audio" or "base64"

def setup_tts_routes(tts_service):
    """Setup TTS routes with the provided TTS service"""
    router = APIRouter(prefix="/api/tts", tags=["tts"])

    @router.get("/stats")
    async def get_tts_stats():
        """Get TTS service statistics"""
        try:
            return await run_in_threadpool(tts_service.get_stats)
        except Exception as e:
            logger.error(f"Failed to get TTS stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/voices")
    async def get_tts_voices():
        """Local (Kokoro) voices grouped by language for the voice selector."""
        return {"provider": "local", "default": DEFAULT_KOKORO_VOICE, "groups": kokoro_voice_groups()}

    def _synthesis_failed():
        reason = ""
        try:
            reason = tts_service.failure_reason()
        except Exception:
            pass
        message = f"Synthesis failed: {reason}" if reason else "Synthesis failed"
        return HTTPException(status_code=500, detail={"message": message})

    @router.post("/synthesize")
    async def synthesize_speech(request: TTSRequest):
        """Synthesize speech from text"""
        try:
            # Synthesis (GPU Kokoro / HTTP endpoint) and the first availability
            # probe (loads the Kokoro model) block; keep them off the event loop.
            if not await run_in_threadpool(lambda: tts_service.available):
                raise HTTPException(
                    status_code=503,
                    detail={"message": "TTS service not available"}
                )
            
            if request.format == "base64":
                audio_b64 = await run_in_threadpool(tts_service.synthesize_to_base64, request.text)
                if not audio_b64:
                    raise _synthesis_failed()
                return {"audio": audio_b64}
            
            else:  # audio format
                audio_data = await run_in_threadpool(tts_service.synthesize, request.text)
                if not audio_data:
                    raise _synthesis_failed()
                
                # Detect format from magic bytes (MP3: ID3 tag or sync word ff e0+)
                is_mp3 = audio_data[:3] == b'ID3' or (len(audio_data) >= 2 and audio_data[0] == 0xff and (audio_data[1] & 0xe0) == 0xe0)
                mime = "audio/mpeg" if is_mp3 else "audio/wav"
                return Response(
                    content=audio_data,
                    media_type=mime,
                    headers={
                        "Content-Disposition": "inline; filename=speech.mp3" if "mpeg" in mime else "inline; filename=speech.wav"
                    }
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"message": f"Synthesis failed: {str(e)}"}
            )

    @router.post("/clear-cache")
    async def clear_tts_cache():
        """Clear TTS cache"""
        try:
            tts_service.clear_cache()
            return {"success": True, "message": "Cache cleared"}
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
