from .whisper_llm import WhisperLLM
from .seamless_m4t import SeamlessM4T
from .qwen_audio import QwenAudio
from typing import Union, Optional, Literal

ModelType = Literal["whisper", "seamless_m4t", "qwen_audio"]

def load_model(
    model_type: ModelType,
    variant: Optional[str] = None,
    device: Optional[str] = None
) -> Union[WhisperLLM, SeamlessM4T, QwenAudio]:
    """
    Load an audio LLM model.
    
    Args:
        model_type: Type of model to load ("whisper", "seamless_m4t", or "qwen_audio")
        variant: Specific variant of the model to load
            - For whisper: "tiny", "base", "small", "medium", "large"
            - For seamless_m4t: Model name from Hugging Face
            - For qwen_audio: Model name from Hugging Face
        device: Device to run the model on
        
    Returns:
        Loaded model instance
    """
    if model_type == "whisper":
        variant = variant or "base"
        return WhisperLLM(model_size=variant, device=device)
    
    elif model_type == "seamless_m4t":
        variant = variant or "facebook/seamless-m4t-large"
        return SeamlessM4T(model_name=variant, device=device)
    
    elif model_type == "qwen_audio":
        variant = variant or "Qwen/Qwen-Audio"
        return QwenAudio(model_name=variant, device=device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'whisper', 'seamless_m4t', or 'qwen_audio'") 