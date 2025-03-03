import os
import sys
import argparse
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.audio_to_text import load_model

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using different models")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, default="whisper", choices=["whisper", "wav2vec2", "hubert"], 
                        help="Model to use for transcription")
    parser.add_argument("--variant", type=str, default=None, help="Specific model variant")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    print(f"Loading {args.model} model...")
    model = load_model(args.model, variant=args.variant)
    
    print(f"Transcribing {args.audio}...")
    start_time = time.time()
    result = model.transcribe(args.audio)
    elapsed = time.time() - start_time
    
    print("\nTranscription Results:")
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 