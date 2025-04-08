import base64
import json
import os
from openai import OpenAI
import traceback
import argparse
import glob
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime
import logging
import sys

class AudioEmotionEvaluator:
    def __init__(self, log_file: str = None, api_key: str = None):
        """
        Initialize the evaluator
        
        Args:
            log_file: Path to log file (optional)
            api_key: OpenAI API key (optional, will try environment variable if not provided)
        """
        # Initialize logger
        self.logger = logging.getLogger('AudioEmotionEvaluator')
        self.logger.setLevel(logging.INFO)
        
        # Create log file with timestamp if not provided
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'emotion_eval_{timestamp}.log'
        
        # Add file handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Initialize paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Emotion label mapping
        self.emotion_mapping = {
            "Neutral state": "neutral",
            "Frustration": "frustrated",
            "Anger": "angry",
            "Happiness": "happy",
            "Excited": "happy",  # Merge excited with happy
            "Sadness": "sad",
            "Fear": "anxious",
            "Surprise": "surprised",
            "Disgust": "angry",  # Map disgust to angry
            "Other": "neutral"   # Default mapping
        }
        
        # Initialize OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it either through the api_key parameter "
                "or by setting the OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.logger.info("OpenAI client initialized successfully")
        
        # Initialize GPT response storage
        self.gpt_responses = []

    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute path"""
        if os.path.isabs(relative_path):
            return relative_path
            
        # If path starts with ../, parse from project root
        if relative_path.startswith("../"):
            # Remove leading ../ and build path from project root
            rel_path = relative_path.replace("../", "", 1)
            abs_path = os.path.join(self.project_root, rel_path)
        else:
            # Otherwise parse from script directory
            abs_path = os.path.join(self.script_dir, relative_path)
            
        # Normalize path
        normalized_path = os.path.normpath(abs_path)
        self.logger.info(f"Path conversion: {relative_path} -> {normalized_path}")
        return normalized_path

    def encode_audio(self, audio_file_path: str) -> str:
        """Encode audio file to base64 string"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            return base64.b64encode(audio_data).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Audio encoding error: {e}")
            raise

    def detect_emotion(self, audio_base64: str) -> str:
        """Detect emotion from audio using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-audio-preview",
                messages=[
                    {"role": "system", "content": "You are an emotion detection assistant. Analyze the audio and determine the primary emotion expressed. Only respond with a single emotion word from this list: Neutral state, Frustration, Anger, Happiness, Excited, Sadness, Fear, Surprise, Disgust."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is the primary emotion expressed in this audio?"},
                        {"type": "input_audio", "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }}
                    ]}
                ]
            )
            
            emotion = response.choices[0].message.content.strip()
            self.logger.info(f"GPT detected emotion: {emotion}")
            
            # Record GPT response
            self.gpt_responses.append({
                "timestamp": datetime.now().isoformat(),
                "response": emotion,
                "raw_response": response.model_dump()
            })
            
            return emotion
            
        except Exception as e:
            self.logger.error(f"Emotion detection error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "Neutral state"

    def map_emotion(self, emotion: str) -> str:
        """Map IEMOCAP emotion labels to standard format"""
        if emotion not in self.emotion_mapping:
            for key, value in self.emotion_mapping.items():
                if key.lower() in emotion.lower():
                    return value
            return "neutral"
        return self.emotion_mapping[emotion]

    def get_majority_emotion(self, emotions: List[str]) -> str:
        """Get majority emotion label from multiple annotations"""
        if not emotions:
            return "unknown"
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts.items(), key=lambda x: x[1])[0]

    def find_audio_file(self, utterance_id: str, iemocap_root: str) -> str:
        """Find audio file path based on utterance ID"""
        abs_iemocap_root = self.get_absolute_path(iemocap_root)
        parts = utterance_id.split('_')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid utterance ID format: {utterance_id}")

        session_id = parts[0]
        if not session_id.startswith('Ses') or len(session_id) < 5:
            raise ValueError(f"Invalid session ID format: {session_id}")
        
        try:
            session_num = int(session_id[3:5])
            if session_num < 1 or session_num > 5:
                raise ValueError(f"Session number out of range (1-5): {session_num}")
        except ValueError as e:
            raise ValueError(f"Cannot parse session number: {session_id}") from e

        session = f"Session{session_num}"
        dialogue = f"{parts[0]}_{parts[1]}"
        audio_filename = f"{utterance_id}.wav"

        session_path = os.path.join(abs_iemocap_root, session)
        
        # Standard path patterns
        patterns = [
            os.path.join(abs_iemocap_root, session, "sentences", "wav", dialogue, audio_filename),
            os.path.join(abs_iemocap_root, session, "wav", dialogue, audio_filename),
            os.path.join(abs_iemocap_root, session, "dialog", "wav", dialogue, audio_filename)
        ]

        # Check standard paths
        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern

        # Recursive search
        for root, _, files in os.walk(session_path):
            if audio_filename in files:
                return os.path.join(root, audio_filename)

        raise FileNotFoundError(f"Audio file not found: {utterance_id}")

    def evaluate_emotions(self, json_file_path: str, max_samples: int, iemocap_root: str) -> Dict:
        """
        Evaluate emotion recognition accuracy
        
        Args:
            json_file_path: Path to the JSON file containing emotion data
            max_samples: Maximum number of samples to evaluate (-1 for all samples)
            iemocap_root: Root directory of IEMOCAP dataset
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle full dataset testing
            if max_samples == -1:
                sample_data = data
                self.logger.info(f"Running full dataset evaluation with {len(data)} samples")
            else:
                sample_data = data[:max_samples]
                self.logger.info(f"Running evaluation with {len(sample_data)} samples")
            
            results = []
            total_samples = 0
            correct_samples = 0
            
            # Process samples with progress bar
            for item in tqdm(sample_data, desc="Processing audio samples", unit="sample"):
                try:
                    utterance_id = item.get('id')
                    if not utterance_id:
                        continue

                    audio_path = self.find_audio_file(utterance_id, iemocap_root)
                    true_emotion = self.get_majority_emotion(item.get('emotion', []))
                    
                    audio_base64 = self.encode_audio(audio_path)
                    predicted_emotion = self.detect_emotion(audio_base64)

                    true_emotion_std = self.map_emotion(true_emotion)
                    predicted_emotion_std = self.map_emotion(predicted_emotion)

                    is_correct = predicted_emotion_std == true_emotion_std
                    if is_correct:
                        correct_samples += 1

                    results.append({
                        "id": utterance_id,
                        "audio_path": audio_path,
                        "true_emotion": true_emotion,
                        "predicted_emotion": predicted_emotion,
                        "true_emotion_mapped": true_emotion_std,
                        "predicted_emotion_mapped": predicted_emotion_std,
                        "is_correct": is_correct
                    })

                    total_samples += 1

                except Exception as e:
                    self.logger.error(f"Error processing sample {item.get('id')}: {str(e)}")
                    self.logger.error(traceback.format_exc())

            accuracy = correct_samples / total_samples if total_samples > 0 else 0

            final_results = {
                "total_evaluated": total_samples,
                "correct_predictions": correct_samples,
                "accuracy": accuracy,
                "detailed_results": results,
                "gpt_responses": self.gpt_responses
            }

            # Save results to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'evaluation_results_{timestamp}.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            return final_results

        except Exception as e:
            self.logger.error(f"Evaluation process error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Audio Emotion Evaluation Tool')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to IEMOCAP JSON file (relative to project root)')
    parser.add_argument('--max_samples', type=int, default=10,
                      help='Maximum number of samples to evaluate (use -1 for full dataset)')
    parser.add_argument('--iemocap_root', type=str, required=True,
                      help='Path to IEMOCAP root directory (relative to project root)')
    parser.add_argument('--log_file', type=str,
                      help='Path to log file (optional)')
    parser.add_argument('--api_key', type=str,
                      help='OpenAI API key (optional, will use environment variable if not provided)')
    
    args = parser.parse_args()

    try:
        evaluator = AudioEmotionEvaluator(args.log_file, args.api_key)
        
        data_path = evaluator.get_absolute_path(args.data)
        iemocap_root = evaluator.get_absolute_path(args.iemocap_root)
        
        # Add information about test scope
        if args.max_samples == -1:
            print("\nStarting full dataset evaluation...")
        else:
            print(f"\nStarting evaluation with {args.max_samples} samples...")
        
        results = evaluator.evaluate_emotions(
            data_path,
            args.max_samples,
            iemocap_root
        )
        
        print("\nEvaluation completed successfully!")
        print(f"Total samples evaluated: {results['total_evaluated']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Detailed results saved to: evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nIf the error is related to the API key, you can:")
        print("1. Set the OPENAI_API_KEY environment variable, or")
        print("2. Provide the API key directly using the --api_key parameter")
        sys.exit(1)

if __name__ == "__main__":
    main() 