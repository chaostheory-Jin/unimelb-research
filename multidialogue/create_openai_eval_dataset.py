import json
import base64
import os
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import sys

def setup_logger(log_file=None):
    """Set up logger"""
    logger = logging.getLogger('OpenAIDatasetGenerator')
    logger.setLevel(logging.INFO)
    
    # Create log file with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'dataset_generation_{timestamp}.log'
    
    # Add file handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Add console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def get_absolute_path(relative_path, project_root):
    """Convert relative path to absolute path"""
    if os.path.isabs(relative_path):
        return relative_path
        
    # If path starts with ../, parse from project root
    if relative_path.startswith("../"):
        # Remove leading ../ and build path from project root
        rel_path = relative_path.replace("../", "", 1)
        abs_path = os.path.join(project_root, rel_path)
    else:
        # Otherwise parse from current directory
        abs_path = os.path.join(os.getcwd(), relative_path)
        
    # Normalize path
    return os.path.normpath(abs_path)

def encode_audio_file(audio_path):
    """Encode audio file to base64"""
    try:
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error encoding audio file {audio_path}: {str(e)}")

def find_audio_file(utterance_id, iemocap_root):
    """Find audio file path based on utterance ID"""
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

    session_path = os.path.join(iemocap_root, session)
    
    # Standard path patterns
    patterns = [
        os.path.join(iemocap_root, session, "sentences", "wav", dialogue, audio_filename),
        os.path.join(iemocap_root, session, "wav", dialogue, audio_filename),
        os.path.join(iemocap_root, session, "dialog", "wav", dialogue, audio_filename)
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

def map_emotion(emotion, emotion_mapping):
    """Map IEMOCAP emotion labels to standard format"""
    if emotion not in emotion_mapping:
        for key, value in emotion_mapping.items():
            if key.lower() in emotion.lower():
                return value
        return "neutral"
    return emotion_mapping[emotion]

def get_majority_emotion(emotions):
    """Get majority emotion label from multiple annotations"""
    if not emotions:
        return "unknown"
    
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return max(emotion_counts.items(), key=lambda x: x[1])[0]

def create_openai_eval_dataset(input_json, output_jsonl, iemocap_root, max_samples=-1, output_format="jsonl"):
    """
    Convert JSON dataset to OpenAI evaluation format with base64 encoded audio
    
    Args:
        input_json: Path to input JSON file
        output_jsonl: Path to output JSONL file
        iemocap_root: Path to IEMOCAP dataset root
        max_samples: Maximum number of samples to process (-1 for all)
        output_format: Output format (jsonl or json)
    """
    logger = setup_logger()
    
    # Emotion mapping for standardization
    emotion_mapping = {
        "Neutral state": "neutral",
        "Frustration": "frustrated",
        "Anger": "angry",
        "Happiness": "happy",
        "Excited": "happy", 
        "Sadness": "sad",
        "Fear": "anxious",
        "Surprise": "surprised",
        "Disgust": "angry",
        "Other": "neutral" 
    }
    
    # Project root directory (for path resolution)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Convert paths to absolute
    abs_input_json = get_absolute_path(input_json, project_root)
    abs_iemocap_root = get_absolute_path(iemocap_root, project_root)
    
    logger.info(f"Input JSON: {abs_input_json}")
    logger.info(f"IEMOCAP root: {abs_iemocap_root}")
    logger.info(f"Output file: {output_jsonl}")
    
    try:
        # Load input JSON
        with open(abs_input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from input JSON")
        
        # Handle max samples
        if max_samples == -1:
            samples_to_process = data
            logger.info(f"Processing all {len(samples_to_process)} samples")
        else:
            samples_to_process = data[:max_samples]
            logger.info(f"Processing {len(samples_to_process)} samples")
        
        # Create output data
        output_data = []
        count_success = 0
        count_fail = 0
        
        # Process each sample
        for item in tqdm(samples_to_process, desc="Processing audio samples"):
            try:
                utterance_id = item.get('id')
                if not utterance_id:
                    logger.warning(f"Skipping item with no ID: {item}")
                    count_fail += 1
                    continue
                
                # Find audio file
                audio_path = find_audio_file(utterance_id, abs_iemocap_root)
                
                # Get ground truth emotion
                true_emotion = get_majority_emotion(item.get('emotion', []))
                true_emotion_std = map_emotion(true_emotion, emotion_mapping)
                
                # Encode audio to base64
                audio_base64 = encode_audio_file(audio_path)
                
                # Create sample in OpenAI evaluation format
                sample = {
                    "id": utterance_id,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an emotion detection assistant. Analyze the audio and determine the primary emotion expressed. Only respond with a single emotion word from this list: neutral, frustrated, angry, happy, sad, anxious, surprised."
                        },
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "What is the primary emotion expressed in this audio?"},
                                {"type": "input_audio", "input_audio": {
                                    "data": audio_base64,
                                    "format": "wav"
                                }}
                            ]
                        }
                    ],
                    "ideal_response": true_emotion_std
                }
                
                output_data.append(sample)
                count_success += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {item.get('id')}: {str(e)}")
                count_fail += 1
        
        # Write output based on format
        if output_format.lower() == "jsonl":
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Successfully wrote {len(output_data)} samples to JSONL file: {output_jsonl}")
        else:
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully wrote {len(output_data)} samples to JSON file: {output_jsonl}")
        
        logger.info(f"Processing summary: {count_success} successful, {count_fail} failed")
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Create OpenAI Evaluation Dataset')
    parser.add_argument('--input', type=str, required=True,
                      help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True,
                      help='Output JSONL file path')
    parser.add_argument('--iemocap_root', type=str, required=True,
                      help='IEMOCAP dataset root directory')
    parser.add_argument('--max_samples', type=int, default=-1,
                      help='Maximum number of samples to process (-1 for all)')
    parser.add_argument('--format', type=str, default="jsonl", choices=["jsonl", "json"],
                      help='Output format (jsonl or json)')
    
    args = parser.parse_args()
    
    create_openai_eval_dataset(
        args.input,
        args.output,
        args.iemocap_root,
        args.max_samples,
        args.format
    )

if __name__ == "__main__":
    main() 