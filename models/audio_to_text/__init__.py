from .whisper_model import WhisperModel, WhisperTranscription
from .wav2vec2_model import Wav2Vec2Model, Wav2Vec2Transcription
from .hubert_model import HubertModel, HubertTranscription
from typing import Union, Optional, Literal

ModelType = Literal["whisper", "wav2vec2", "hubert"]
TranscriptionType = Union[WhisperTranscription, Wav2Vec2Transcription, HubertTranscription]

def load_model(
    model_type: ModelType,
    variant: Optional[str] = None,
    device: Optional[str] = None
) -> Union[WhisperModel, Wav2Vec2Model, HubertModel]:
    """
    Load an audio-to-text model.
    
    Args:
        model_type: Type of model to load ("whisper", "wav2vec2", or "hubert")
        variant: Specific variant of the model to load
            - For whisper: "tiny", "base", "small", "medium", "large"
            - For wav2vec2: Model name from Hugging Face
            - For hubert: Model name from Hugging Face
        device: Device to run the model on
        
    Returns:
        Loaded model instance
    """
    if model_type == "whisper":
        variant = variant or "base"
        return WhisperModel(model_size=variant, device=device)
    
    elif model_type == "wav2vec2":
        variant = variant or "facebook/wav2vec2-base-960h"
        return Wav2Vec2Model(model_name=variant, device=device)
    
    elif model_type == "hubert":
        variant = variant or "facebook/hubert-large-ls960-ft"
        return HubertModel(model_name=variant, device=device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'whisper', 'wav2vec2', or 'hubert'") 