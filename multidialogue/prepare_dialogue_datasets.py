import os
import json
import base64
import argparse
from typing import Dict, List
from datetime import datetime
import logging
import traceback
from tqdm import tqdm

class DialogueDatasetPreparer:
    """
    Prepare IEMOCAP dialogue dataset for both experimental and control groups:
    - Experimental group: Single emotion slice for response generation
    - Control group: Full dialogue context for response generation
    """
    
    def __init__(self, iemocap_data_file: str, log_file: str = None):
        """
        Initialize the dataset preparer
        
        Args:
            iemocap_data_file: Path to the processed IEMOCAP JSON file
            log_file: Path to log file (optional)
        """
        # Setup logger
        self.logger = self._setup_logger(log_file)
        
        # Store file paths
        self.iemocap_data_file = iemocap_data_file
        
        # Initialize paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Emotion mapping (from standard IEMOCAP format to our simplified format)
        self.emotion_mapping = {
            "angry": "angry",
            "Anger": "angry",
            "frustrated": "frustrated",
            "Frustration": "frustrated",
            "happy": "happy", 
            "Happiness": "happy",
            "excited": "happy",
            "Excited": "happy",
            "sad": "sad",
            "Sadness": "sad",
            "neutral": "neutral",
            "Neutral state": "neutral",
            "surprised": "surprised",
            "Surprise": "surprised",
            "fear": "anxious",
            "Fear": "anxious",
            "disgust": "angry",
            "Disgust": "angry",
            "other": "neutral",
            "Other": "neutral",
            "unknown": "neutral",
            "Unknown": "neutral"
        }
        
        # Load the dialogue data
        self.data = self._load_dialogue_data()
        
        # Group utterances by dialogue ID
        self.dialogue_groups = self._group_by_dialogue()
        
    def _setup_logger(self, log_file: str = None) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('DialogueDatasetPreparer')
        logger.setLevel(logging.INFO)
        
        # Create log file with timestamp if not provided
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'dialogue_prep_{timestamp}.log'
        
        # Add file handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
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
    
    def _load_dialogue_data(self) -> List[Dict]:
        """Load dialogue data from JSON file"""
        try:
            file_path = self.get_absolute_path(self.iemocap_data_file)
            self.logger.info(f"Loading dialogue data from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Log basic information about loaded data
            if isinstance(data, list):
                self.logger.info(f"Loaded {len(data)} utterances")
                
                # Sample a few items to understand the structure
                if len(data) > 0:
                    sample = data[0]
                    self.logger.info(f"Sample utterance structure: {list(sample.keys())}")
                    
                return data
            else:
                self.logger.error(f"Expected list but got {type(data)}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error loading dialogue data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []
    
    def _group_by_dialogue(self) -> Dict[str, List[Dict]]:
        """Group utterances by dialogue ID"""
        dialogue_groups = {}
        
        try:
            # First, check if we have utterance IDs in the data
            if not self.data or 'id' not in self.data[0]:
                self.logger.error("Data does not contain utterance IDs")
                return dialogue_groups
                
            # Process each utterance
            for utterance in self.data:
                utterance_id = utterance.get('id', '')
                if not utterance_id:
                    continue
                    
                # Extract dialogue ID from utterance ID 
                # Format in IEMOCAP: Ses01F_impro01_M001 -> Ses01F_impro01
                parts = utterance_id.split('_')
                if len(parts) >= 2:
                    dialogue_id = f"{parts[0]}_{parts[1]}"
                    
                    if dialogue_id not in dialogue_groups:
                        dialogue_groups[dialogue_id] = []
                        
                    dialogue_groups[dialogue_id].append(utterance)
            
            # Sort utterances in each dialogue by their ID
            for dialogue_id, utterances in dialogue_groups.items():
                utterances.sort(key=lambda x: x['id'])
                
            self.logger.info(f"Grouped utterances into {len(dialogue_groups)} dialogues")
            
            # Sample a dialogue to log its structure
            if dialogue_groups:
                sample_id = next(iter(dialogue_groups))
                sample_dialogue = dialogue_groups[sample_id]
                self.logger.info(f"Sample dialogue {sample_id}: {len(sample_dialogue)} utterances")
                
            return dialogue_groups
            
        except Exception as e:
            self.logger.error(f"Error grouping dialogues: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def map_emotion(self, emotion: str) -> str:
        """Map emotion to standardized format"""
        return self.emotion_mapping.get(emotion, "neutral")
    
    def prepare_experimental_dataset(self, output_file: str, max_samples: int = -1) -> Dict:
        """
        Prepare dataset for the experimental group (individual slices)
        
        Args:
            output_file: Output file path
            max_samples: Maximum number of samples (-1 for all)
            
        Returns:
            Dataset statistics
        """
        try:
            experimental_data = []
            
            # Process each dialogue
            for dialogue_id, utterances in tqdm(self.dialogue_groups.items(), desc="Processing dialogues for experimental dataset"):
                for i, utterance in enumerate(utterances):
                    # Skip the first utterance as we need at least one previous context
                    if i == 0:
                        continue
                        
                    # Get current and previous utterance
                    prev_utterance = utterances[i-1]
                    
                    # Extract speaker information
                    speaker = "M" if "_M" in utterance['id'] else "F"
                    prev_speaker = "M" if "_M" in prev_utterance['id'] else "F"
                    
                    # Extract or generate reference text
                    reference_text = utterance.get('text', '')
                    
                    # Construct sample with just the previous utterance as context
                    sample = {
                        "id": utterance.get('id', f"{dialogue_id}_{i}"),
                        "audio_path": utterance.get('audio_path', ""),
                        "text": utterance.get('text', ""),
                        "emotion": [self.map_emotion(e) for e in utterance.get('emotion', ["neutral"])],
                        "context": [
                            {
                                "text": prev_utterance.get('text', ""),
                                "speaker": prev_speaker,
                                "is_assistant": False
                            }
                        ],
                        "reference": reference_text
                    }
                    
                    experimental_data.append(sample)
            
            # Limit samples if specified
            if max_samples > 0 and max_samples < len(experimental_data):
                experimental_data = experimental_data[:max_samples]
                
            # Save to file
            output_path = self.get_absolute_path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(experimental_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved {len(experimental_data)} experimental samples to {output_path}")
            
            return {
                "total_samples": len(experimental_data),
                "dialogues": len(self.dialogue_groups)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing experimental dataset: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def prepare_control_dataset(self, output_file: str, max_samples: int = -1) -> Dict:
        """
        Prepare dataset for the control group (full context)
        
        Args:
            output_file: Output file path
            max_samples: Maximum number of samples (-1 for all)
            
        Returns:
            Dataset statistics
        """
        try:
            control_data = []
            
            # Process each dialogue
            for dialogue_id, utterances in tqdm(self.dialogue_groups.items(), desc="Processing dialogues for control dataset"):
                for i, utterance in enumerate(utterances):
                    # Skip the first utterance as we need at least one previous context
                    if i == 0:
                        continue
                        
                    # Get all previous utterances as context
                    prev_utterances = utterances[:i]
                    
                    # Construct full context
                    context = []
                    for prev in prev_utterances:
                        speaker = "M" if "_M" in prev['id'] else "F"
                        context.append({
                            "text": prev.get('text', ""),
                            "speaker": speaker,
                            "is_assistant": False
                        })
                    
                    # Construct sample with full context
                    sample = {
                        "id": utterance.get('id', f"{dialogue_id}_{i}"),
                        "audio_path": utterance.get('audio_path', ""),
                        "text": utterance.get('text', ""),
                        "emotion": [self.map_emotion(e) for e in utterance.get('emotion', ["neutral"])],
                        "context": context,
                        "reference": utterance.get('text', "")
                    }
                    
                    control_data.append(sample)
            
            # Limit samples if specified
            if max_samples > 0 and max_samples < len(control_data):
                control_data = control_data[:max_samples]
                
            # Save to file
            output_path = self.get_absolute_path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(control_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved {len(control_data)} control samples to {output_path}")
            
            return {
                "total_samples": len(control_data),
                "dialogues": len(self.dialogue_groups)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing control dataset: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Dialogue Dataset Preparation Tool')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to IEMOCAP processed JSON file')
    parser.add_argument('--max_samples', type=int, default=-1,
                      help='Maximum number of samples to include (-1 for all)')
    parser.add_argument('--experimental_output', type=str, default="experimental_dataset.json",
                      help='Output file for experimental dataset (single slice context)')
    parser.add_argument('--control_output', type=str, default="control_dataset.json",
                      help='Output file for control dataset (full dialogue context)')
    parser.add_argument('--log_file', type=str,
                      help='Path to log file (optional)')
    
    args = parser.parse_args()

    try:
        preparer = DialogueDatasetPreparer(args.data, args.log_file)
        
        # Prepare experimental dataset
        print("\nPreparing experimental dataset (single slice context)...")
        exp_stats = preparer.prepare_experimental_dataset(
            args.experimental_output,
            args.max_samples
        )
        
        # Prepare control dataset
        print("\nPreparing control dataset (full dialogue context)...")
        ctrl_stats = preparer.prepare_control_dataset(
            args.control_output,
            args.max_samples
        )
        
        print("\nDataset preparation completed!")
        print(f"Experimental dataset: {exp_stats.get('total_samples', 0)} samples")
        print(f"Control dataset: {ctrl_stats.get('total_samples', 0)} samples")
        print(f"Unique dialogues: {exp_stats.get('dialogues', 0)}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()