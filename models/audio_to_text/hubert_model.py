import os
import torch
import librosa
import numpy as np
from transformers import HubertForCTC, Wav2Vec2Processor
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class HubertTranscription:
    text: str
    confidence: float
    tokens: Optional[List[str]] = None
    token_probs: Optional[List[float]] = None

class HubertModel:
    """
    Implementation of Facebook AI's HuBERT model for speech recognition.
    
    HuBERT (Hidden-Unit BERT) is a self-supervised speech representation model
    that uses masked prediction of continuous speech units.
    """
    
    def __init__(self, model_name: str = "facebook/hubert-large-ls960-ft", device: Optional[str] = None):
        """
        Initialize the HuBERT model.
        
        Args:
            model_name: Name or path of the model to use
            device: Device to run the model on. If None, will use CUDA if available, else CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertForCTC.from_pretrained(model_name).to(device)
        print(f"Loaded HuBERT model {model_name} on {device}")
    
    def transcribe(self, audio_path: str, return_token_details: bool = False) -> HubertTranscription:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            return_token_details: Whether to return token-level details
            
        Returns:
            HubertTranscription object containing the transcription results
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
        
        result = HubertTranscription(
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
        return f"HubertModel(model={self.model_name}, device={self.device})" 