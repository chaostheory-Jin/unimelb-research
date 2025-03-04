import os
import sys
import argparse
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.audio_llm import load_model

def main():
    parser = argparse.ArgumentParser(description="Process audio with end-to-end audio LLMs")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, default="whisper", 
                        choices=["whisper", "seamless_m4t", "qwen_audio"], 
                        help="Model to use for processing")
    parser.add_argument("--variant", type=str, default=None, help="Specific model variant")
    parser.add_argument("--prompt", type=str, default=None, help="Optional text prompt")
    parser.add_argument("--target_lang", type=str, default="eng", help="Target language (for seamless_m4t)")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    print(f"Loading {args.model} model...")
    model = load_model(args.model, variant=args.variant)
    
    print(f"Processing {args.audio}...")
    start_time = time.time()
    
    if args.model == "whisper":
        result = model.generate(args.audio, prompt=args.prompt)
    elif args.model == "seamless_m4t":
        result = model.generate(args.audio, target_language=args.target_lang)
    elif args.model == "qwen_audio":
        result = model.generate(args.audio, prompt=args.prompt)
    
    elapsed = time.time() - start_time
    
    print("\nResults:")
    print(f"Generated text: {result['text']}")
    for key, value in result.items():
        if key != "text" and key != "segments":
            print(f"{key}: {value}")
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 