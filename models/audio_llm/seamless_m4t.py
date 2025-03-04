import os
import torch
from transformers import AutoProcessor, SeamlessM4TModel
from typing import Optional, Dict, Any, List, Union

class SeamlessM4T:
    """
    Implementation of Meta's Seamless M4T (Massively Multilingual & Multimodal Machine Translation) model.
    
    Seamless M4T can directly process speech in 100+ languages and generate text
    in 100+ languages, supporting speech-to-text, speech-to-speech, text-to-text,
    and text-to-speech tasks.
    """
    
    def __init__(self, model_name: str = "facebook/seamless-m4t-large", device: Optional[str] = None):
        """
        Initialize the Seamless M4T model.
        
        Args:
            model_name: Name or path of the model to use
            device: Device to run the model on. If None, will use CUDA if available, else CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4TModel.from_pretrained(model_name).to(device)
        print(f"Loaded Seamless M4T model {model_name} on {device}")
    
    def generate(self, audio_path: str, target_language: str = "eng", 
                 src_language: Optional[str] = None, 
                 task: str = "s2tt", **kwargs) -> Dict[str, Any]:
        """
        Process audio and generate text using the Seamless M4T model.
        
        Args:
            audio_path: Path to the audio file
            target_language: Target language code (e.g., "eng", "fra", "cmn")
            src_language: Source language code (optional, model can auto-detect)
            task: Task to perform ("s2tt" for speech-to-text translation)
            **kwargs: Additional arguments to pass to model.generate
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        import librosa
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load and preprocess audio
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Prepare inputs
        inputs = self.processor(
            audios=audio_array,
            src_lang=src_language,
            tgt_lang=target_language,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        output_tokens = self.model.generate(
            **inputs,
            tgt_lang=target_language,
            generate_speech=False,
            **kwargs
        )
        
        # Decode output
        transcription = self.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        
        return {
            "text": transcription,
            "target_language": target_language,
            "source_language": src_language or "auto-detected",
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
        return f"SeamlessM4T(model={self.model_name}, device={self.device})" 