import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from typing import Optional, Dict, Any, List, Union

class QwenAudio:
    """
    Implementation of Qwen Audio, a multimodal LLM that can process audio inputs.
    
    Qwen Audio can understand and respond to audio content, supporting tasks like
    audio transcription, audio understanding, and audio-based conversation.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen-Audio", device: Optional[str] = None):
        """
        Initialize the Qwen Audio model.
        
        Args:
            model_name: Name or path of the model to use
            device: Device to run the model on. If None, will use CUDA if available, else CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        print(f"Loaded Qwen Audio model {model_name} on {device}")
    
    def generate(self, audio_path: str, prompt: Optional[str] = None, 
                max_new_tokens: int = 512, temperature: float = 0.7, 
                top_p: float = 0.9, **kwargs) -> Dict[str, Any]:
        """
        Process audio and generate text using the Qwen Audio model.
        
        Args:
            audio_path: Path to the audio file
            prompt: Optional text prompt to guide the generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
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
        if prompt:
            query = f"{prompt}\n[AUDIO]"
        else:
            query = "[AUDIO]"
        
        inputs = self.processor(
            text=query,
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        
        # Decode output
        response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        
        # Extract the model's response (after the query)
        if query in response:
            response = response.split(query)[1].strip()
        
        return {
            "text": response,
            "prompt": prompt,
            "audio_path": audio_path
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
        return f"QwenAudio(model={self.model_name}, device={self.device})" 