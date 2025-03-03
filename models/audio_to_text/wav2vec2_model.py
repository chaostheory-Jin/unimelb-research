import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class Wav2Vec2Transcription:
    text: str
    confidence: float
    tokens: Optional[List[str]] = None
    token_probs: Optional[List[float]] = None

class Wav2Vec2Model:
    """
    Implementation of Facebook AI's Wav2Vec2 model for speech recognition.
    
    Wav2Vec2 is a self-supervised learning framework for speech recognition
    that learns representations from raw audio data.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: Optional[str] = None):
        """
        Initialize the Wav2Vec2 model.
        
        Args:
            model_name: Name or path of the model to use
            device: Device to run the model on. If None, will use CUDA if available, else CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        print(f"Loaded Wav2Vec2 model {model_name} on {device}")
    
    def transcribe(self, audio_path: str, return_token_details: bool = False) -> Wav2Vec2Transcription:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            return_token_details: Whether to return token-level details
            
        Returns:
            Wav2Vec2Transcription object containing the transcription results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load and preprocess audio
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").to(self.device)
        
        # Get model output
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Get predicted tokens and probabilities
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # Calculate confidence score (mean of highest probability per token)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = torch.mean(torch.max(probs, dim=-1).values).item()
        
        result = Wav2Vec2Transcription(
            text=transcription,
            confidence=confidence
        )
        
        if return_token_details:
            tokens = self.processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())
            token_probs = [probs[0, i, id].item() for i, id in enumerate(predicted_ids[0])]
            result.tokens = tokens
            result.token_probs = token_probs
        
        return result
    
    def __str__(self):
        return f"Wav2Vec2Model(model={self.model_name}, device={self.device})" 