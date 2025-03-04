import os
import torch
import whisper
from typing import Optional, Dict, Any, List

class WhisperLLM:
    """
    Implementation of OpenAI's Whisper model as an end-to-end audio LLM.
    
    Whisper can directly process audio and generate text transcriptions
    with multilingual support and additional capabilities like translation.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the Whisper LLM.
        
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
    
    def generate(self, audio_path: str, prompt: Optional[str] = None, 
                task: str = "transcribe", language: Optional[str] = None,
                temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Process audio and generate text using the Whisper model.
        
        Args:
            audio_path: Path to the audio file
            prompt: Optional text prompt to guide the generation
            task: Task to perform ("transcribe" or "translate")
            language: Language code (e.g., "en", "fr", "zh")
            temperature: Sampling temperature (0.0 means deterministic)
            **kwargs: Additional arguments to pass to whisper.transcribe
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        options = {
            "task": task,
            "temperature": temperature,
        }
        
        if prompt:
            options["prompt"] = prompt
        
        if language:
            options["language"] = language
        
        options.update(kwargs)
        
        result = self.model.transcribe(audio_path, **options)
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result.get("language", ""),
            "task": task
        }
    
    def batch_generate(self, audio_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple audio files and generate text for each.
        
        Args:
            audio_paths: List of paths to audio files
            **kwargs: Additional arguments to pass to generate
            
        Returns:
            List of dictionaries containing the generated text and metadata
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.generate(audio_path, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "audio_path": audio_path})
        
        return results
    
    def __str__(self):
        return f"WhisperLLM(size={self.model_size}, device={self.device})" 