import os
import torch
import whisper
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

@dataclass
class WhisperTranscription:
    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float

class WhisperModel:
    """
    Implementation of OpenAI's Whisper model for speech recognition.
    
    Whisper is a general-purpose speech recognition model that can transcribe
    speech in multiple languages and translate it to English.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the Whisper model.
        
        Args:
            model_size: Size of the model to use. Options: "tiny", "base", "small", "medium", "large"
            device: Device to run the model on. If None, will use CUDA if available, else CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_size = model_size
        self.model = whisper.load_model(model_size, device=device)
        print(f"Loaded Whisper {model_size} model on {device}")
    
    def transcribe(self, audio_path: str, **kwargs) -> WhisperTranscription:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional arguments to pass to whisper.transcribe
            
        Returns:
            WhisperTranscription object containing the transcription results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = self.model.transcribe(audio_path, **kwargs)
        
        return WhisperTranscription(
            text=result["text"],
            segments=result["segments"],
            language=result.get("language", ""),
            confidence=result.get("confidence", 0.0)
        )
    
    def __str__(self):
        return f"WhisperModel(size={self.model_size}, device={self.device})" 