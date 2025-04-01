import os
import json
import re
from pydub import AudioSegment
import glob
from typing import Dict, List, Tuple
from tqdm import tqdm

class DialogueProcessor:
    def __init__(self, iemocap_root: str, output_dir: str):
        """
        Initialize the dialogue processor
        
        Args:
            iemocap_root: Root directory of IEMOCAP dataset
            output_dir: Directory to save processed audio segments and JSON index
        """
        self.iemocap_root = iemocap_root
        self.output_dir = output_dir
        self.processed_data = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for processed audio segments
        self.segments_dir = os.path.join(output_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)

    def parse_timestamp(self, timestamp: str) -> float:
        """Convert timestamp string to seconds"""
        try:
            return float(timestamp)
        except ValueError:
            return 0.0

    def parse_evaluation_file(self, eval_file: str) -> List[Dict]:
        """Parse emotion evaluation file and extract utterance information"""
        utterances = []
        current_dialogue = os.path.basename(eval_file).replace(".txt", "")
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Match lines containing emotion evaluations
                # Format: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me. [XXX]
                match = re.match(r'(\w+)\s+\[(\d+\.\d+)-(\d+\.\d+)\]:\s*(.*?)\s*(?:\[(\w+)\])?$', line.strip())
                if match:
                    utterance_id, start, end, text, emotion = match.groups()
                    
                    utterances.append({
                        "id": utterance_id,
                        "start_time": self.parse_timestamp(start),
                        "end_time": self.parse_timestamp(end),
                        "text": text,
                        "emotion": emotion if emotion else "unknown",
                        "dialogue_id": current_dialogue
                    })
        
        return utterances

    def extract_audio_segment(self, audio_file: str, start_time: float, end_time: float, 
                            output_path: str) -> bool:
        """Extract audio segment from dialogue audio file"""
        try:
            audio = AudioSegment.from_wav(audio_file)
            # Convert times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            # Extract segment
            segment = audio[start_ms:end_ms]
            # Export segment
            segment.export(output_path, format="wav")
            return True
        except Exception as e:
            print(f"Error extracting segment from {audio_file}: {str(e)}")
            return False

    def process_session(self, session_num: int):
        """Process all dialogues in a session"""
        session = f"Session{session_num}"
        eval_dir = os.path.join(self.iemocap_root, session, "dialog", "EmoEvaluation")
        wav_dir = os.path.join(self.iemocap_root, session, "dialog", "wav")
        
        # Find all evaluation files
        eval_files = glob.glob(os.path.join(eval_dir, "*.txt"))
        
        for eval_file in tqdm(eval_files, desc=f"Processing {session}"):
            # Parse evaluation file
            utterances = self.parse_evaluation_file(eval_file)
            
            # Get corresponding audio file
            dialogue_id = os.path.basename(eval_file).replace(".txt", "")
            audio_file = os.path.join(wav_dir, f"{dialogue_id}.wav")
            
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                continue
            
            # Process each utterance
            for utterance in utterances:
                # Create output path for segment
                segment_path = os.path.join(self.segments_dir, 
                                         f"{utterance['id']}.wav")
                
                # Extract audio segment
                if self.extract_audio_segment(audio_file, 
                                           utterance['start_time'],
                                           utterance['end_time'],
                                           segment_path):
                    # Add to processed data
                    self.processed_data.append({
                        "id": utterance['id'],
                        "audio": os.path.relpath(segment_path, self.output_dir),
                        "text": utterance['text'],
                        "emotion": [utterance['emotion']],
                        "start_time": utterance['start_time'],
                        "end_time": utterance['end_time'],
                        "dialogue_id": utterance['dialogue_id']
                    })

    def process_all_sessions(self):
        """Process all sessions in the dataset"""
        for session_num in range(1, 6):  # Sessions 1-5
            self.process_session(session_num)
            
        # Save index to JSON file
        output_json = os.path.join(self.output_dir, "dialog_segments_index.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process IEMOCAP dialogue dataset')
    parser.add_argument('--iemocap_root', type=str, required=True,
                        help='Root directory of IEMOCAP dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    processor = DialogueProcessor(args.iemocap_root, args.output_dir)
    processor.process_all_sessions()

if __name__ == "__main__":
    main() 